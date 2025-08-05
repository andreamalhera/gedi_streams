import random
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Generator, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import heapq
import numpy as np
from collections import defaultdict, deque
import uuid

# --- Constants ---
MAX_CONCURRENT_CASES: int = 10
MAX_DELAY_MINUTES: int = 30
MAX_FRACTAL_DEPTH: int = 3
MAX_GLOBAL_MEMORY: int = 1000
CASE_HISTORY_MAXLEN: int = 100
ACTIVITY_PREFIXES: List[str] = [
    'Process', 'Review', 'Analyze', 'Check', 'Validate', 'Submit', 'Approve', 
    'Execute', 'Monitor', 'Report', 'Update', 'Create', 'Verify', 'Send', 
    'Receive', 'Transform', 'Calculate'
]
ACTIVITY_SUFFIXES: List[str] = [
    'Data', 'Request', 'Document', 'Form', 'Report', 'Status', 'Record', 
    'File', 'Item', 'Task', 'Order', 'Invoice', 'Profile', 'Account', 
    'Transaction', 'Message'
]

# --- Enums and Data Classes ---

class EventType(Enum):
    """Defines the type of an event."""
    START = "start"
    COMPLETE = "complete"

@dataclass(order=True)
class Event:
    """Represents a single event in the stream."""
    timestamp: datetime
    case_id: str
    activity: str
    event_type: EventType
    attributes: Dict[str, Any] = field(default_factory=dict, compare=False)
    process_id: str = field(default="", compare=False)
    parent_case_id: Optional[str] = field(default=None, compare=False)
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()), compare=False)

@dataclass(order=True)
class ScheduledEvent:
    """An event scheduled for future delivery in the simulation."""
    delivery_timestamp: datetime
    event: Event = field(compare=False)

class ProcessTemplate:
    """Defines a reusable process workflow, treated as immutable."""
    def __init__(self, name: str, activities: List[str],
                 dependencies: Dict[str, List[str]],
                 decision_points: Dict[str, List[str]]):
        self.name: str = name
        self.activities: List[str] = activities
        self.dependencies: Dict[str, List[str]] = dependencies
        self.decision_points: Dict[str, List[str]] = decision_points
        self.start_activities: List[str] = [
            act for act in activities if not dependencies.get(act)
        ]

class Case:
    """Represents a single, stateful instance of a process."""
    def __init__(self, case_id: str, process_template: ProcessTemplate, start_time: datetime):
        self.case_id: str = case_id
        self.template: ProcessTemplate = process_template
        self.completed_activities: Set[str] = set()
        self.running_activities: Set[str] = set()
        self.attributes: Dict[str, Any] = {}
        self.start_time: datetime = start_time
        self.long_term_memory: deque = deque(maxlen=CASE_HISTORY_MAXLEN)
        self.decision_history: List[Tuple[str, str]] = []

    def can_start_activity(self, activity: str) -> bool:
        """Checks if an activity's dependencies are met."""
        if activity in self.running_activities or activity in self.completed_activities:
            return False
        deps: List[str] = self.template.dependencies.get(activity, [])
        return all(dep in self.completed_activities for dep in deps)

    def get_next_activities(self) -> List[str]:
        """Gets all activities that can be started now."""
        available: List[str] = []
        # Check all activities in the template
        for activity in self.template.activities:
            if self.can_start_activity(activity):
                available.append(activity)
        # Check activities added by decisions
        for _, outcome in self.decision_history:
            if self.can_start_activity(outcome):
                available.append(outcome)
        return list(set(available))

# --- Core Simulation Engine ---

class EventStreamGenerator:
    """
    Generates a sophisticated event stream using a discrete-event simulation model.
    This approach correctly handles concurrency, dependencies, and other complex behaviors.
    """
    def __init__(self,
                 temporal_dependency_strength: float = 0.7,
                 long_term_dependency_strength: float = 0.7,
                 non_linear_dependency_strength: float = 0.7,
                 out_of_order_strength: float = 0.3,
                 fractal_behavior_strength: float = 0.2,
                 concurrency_percentage: float = 0.5,
                 seed: Optional[int] = None):

        if seed:
            random.seed(seed)
            np.random.seed(seed)

        self.temporal_dependency_strength: float = max(0, min(1, temporal_dependency_strength))
        self.long_term_dependency_strength: float = max(0, min(1, long_term_dependency_strength))
        self.non_linear_dependency_strength: float = max(0, min(1, non_linear_dependency_strength))
        self.out_of_order_strength: float = max(0, min(1, out_of_order_strength))
        self.fractal_behavior_strength: float = max(0, min(1, fractal_behavior_strength))
        self.concurrency_percentage: float = max(0, min(1, concurrency_percentage))

        self.event_queue: List[ScheduledEvent] = []
        self.active_cases: Dict[str, Case] = {}
        self.global_memory: deque = deque(maxlen=MAX_GLOBAL_MEMORY)
        self.simulation_time: datetime = datetime.now()
        self.case_counter: int = 0
        self.templates: Dict[str, ProcessTemplate] = self._generate_random_process_templates()

    def _schedule_event(self, event: Event, delay: Optional[timedelta] = None) -> None:
        """Schedules an event, applying out-of-order delivery."""
        true_timestamp: datetime = event.timestamp
        delivery_delay_minutes: float = 0.0

        if self.out_of_order_strength > 0 and random.random() < (0.2 * self.out_of_order_strength):
            max_delay: float = MAX_DELAY_MINUTES * self.out_of_order_strength
            delivery_delay_minutes = random.uniform(-max_delay, max_delay)
            event.attributes['out_of_order'] = True
            event.attributes['delivery_delay_minutes'] = round(delivery_delay_minutes, 2)
        
        delivery_timestamp: datetime = true_timestamp + timedelta(minutes=delivery_delay_minutes)
        scheduled_event = ScheduledEvent(delivery_timestamp=delivery_timestamp, event=event)
        heapq.heappush(self.event_queue, scheduled_event)

    def generate_event_stream(self, max_events: int) -> Generator[Event, None, None]:
        """Main simulation loop that generates and yields events."""
        self._start_new_case()
        
        events_yielded: int = 0
        while self.event_queue and events_yielded < max_events:
            scheduled_event = heapq.heappop(self.event_queue)
            
            self.simulation_time = scheduled_event.delivery_timestamp
            event: Event = scheduled_event.event
            
            self._process_event(event)
            
            self.global_memory.append(event)
            yield event
            events_yielded += 1

            if (len(self.active_cases) < MAX_CONCURRENT_CASES and random.random() < 0.1):
                 self._start_new_case()

    def _process_event(self, event: Event) -> None:
        """Handles the logic for a single event and schedules subsequent events."""
        case: Optional[Case] = self.active_cases.get(event.case_id)
        if not case:
            return

        if event.event_type == EventType.START:
            self._handle_activity_start(case, event.activity)
        elif event.event_type == EventType.COMPLETE:
            self._handle_activity_complete(case, event.activity)

    def _handle_activity_start(self, case: Case, activity: str) -> None:
        """Logic for when an activity starts."""
        case.running_activities.add(activity)

        # Apply long-term dependency influence on duration
        long_term_factor: float = 1.0
        if self.long_term_dependency_strength > 0:
            if len(self.global_memory) > 50 and random.random() < self.long_term_dependency_strength:
                past_event: Event = random.choice(self.global_memory)
                if "High-Value" in past_event.activity:
                     long_term_factor = 1.5
        
        duration_minutes: float = random.uniform(1.0, 10.0) * long_term_factor
        completion_time: datetime = self.simulation_time + timedelta(minutes=duration_minutes)

        complete_event = Event(
            case_id=case.case_id,
            activity=activity,
            event_type=EventType.COMPLETE,
            timestamp=completion_time,
            process_id=case.template.name,
            attributes={'duration_minutes': round(duration_minutes, 2)}
        )
        self._schedule_event(complete_event)

    def _handle_activity_complete(self, case: Case, activity: str) -> None:
        """Logic for when an activity completes."""
        case.running_activities.discard(activity)
        case.completed_activities.add(activity)
        case.long_term_memory.append(activity)

        # --- Non-Linear Dependencies (Decision Points) ---
        decision_outcome: Optional[str] = self._apply_non_linear_decision(case, activity)
        if decision_outcome:
            # The decision itself doesn't create an event, but enables a new activity path
            pass

        # --- Fractal Behavior ---
        if self.fractal_behavior_strength > 0 and random.random() < (0.1 * self.fractal_behavior_strength):
            self._generate_fractal_subprocess(case, activity, self.simulation_time)

        # --- Schedule next activities ---
        next_activities: List[str] = case.get_next_activities()
        allow_concurrency: bool = random.random() < self.concurrency_percentage
        
        for next_act in next_activities:
            if not allow_concurrency and any(case.running_activities):
                break 
            self._schedule_start_event(case, next_act)

        # --- Case completion check ---
        if not case.running_activities and not case.get_next_activities():
            del self.active_cases[case.case_id]

    def _schedule_start_event(self, case: Case, activity: str) -> None:
        """Schedules a new activity start event with appropriate delays."""
        # Temporal dependency: delay is influenced by recent case activity
        temporal_delay_minutes: float = random.uniform(0.5, 5.0)
        if self.temporal_dependency_strength > 0 and case.long_term_memory:
            recent_complexity = len(set(list(case.long_term_memory)[-5:]))
            temporal_factor = 1 + (recent_complexity * 0.2 * self.temporal_dependency_strength)
            temporal_delay_minutes *= temporal_factor
        
        start_time: datetime = self.simulation_time + timedelta(minutes=temporal_delay_minutes)
        
        start_event = Event(
            case_id=case.case_id,
            activity=activity,
            event_type=EventType.START,
            timestamp=start_time,
            process_id=case.template.name,
            attributes={
                'temporal_delay_minutes': round(temporal_delay_minutes, 2),
                'concurrent_activities_in_case': len(case.running_activities)
            }
        )
        self._schedule_event(start_event)

    def _start_new_case(self) -> None:
        """Initializes a new case and schedules its first event(s)."""
        self.case_counter += 1
        case_id: str = f"case_{self.case_counter:06d}"
        template: ProcessTemplate = random.choice(list(self.templates.values()))
        case = Case(case_id, template, self.simulation_time)
        self.active_cases[case_id] = case

        for activity in case.template.start_activities:
            self._schedule_start_event(case, activity)

    def _apply_non_linear_decision(self, case: Case, activity: str) -> Optional[str]:
        """Models complex decision logic, adding new activities to the potential path."""
        if activity not in case.template.decision_points:
            return None
        
        if self.non_linear_dependency_strength > 0 and random.random() < self.non_linear_dependency_strength:
            options: List[str] = case.template.decision_points[activity]
            
            history_factor: float = (len(case.completed_activities) % 5) / 4.0
            global_factor: float = (len(self.global_memory) % 10) / 9.0
            
            decision_score: float = (
                history_factor * 0.4 * self.non_linear_dependency_strength +
                global_factor * 0.3 * self.non_linear_dependency_strength +
                random.random() * (1 - self.non_linear_dependency_strength * 0.7)
            )
            
            option_index: int = int(decision_score * len(options)) % len(options)
            decision_outcome: str = options[option_index]
            
            # IMPORTANT: Do not modify the template. Track the decision in the case.
            case.decision_history.append((activity, decision_outcome))
            return decision_outcome
        
        return None

    def _generate_fractal_subprocess(self, parent_case: Case, parent_activity: str, start_time: datetime, depth: int = 0) -> None:
        """Generates a smaller, self-similar subprocess and schedules its events."""
        if depth >= MAX_FRACTAL_DEPTH:
            return

        subprocess_id: str = f"{parent_case.case_id}_fractal_{depth+1}"
        fractal_activities: List[str] = list(parent_case.completed_activities)[:2]
        if not fractal_activities:
            return

        # Create a tiny case for the subprocess
        sub_template: ProcessTemplate = self.templates['Micro_Process']
        sub_case = Case(subprocess_id, sub_template, start_time)
        sub_case.parent_case_id = parent_case.case_id
        self.active_cases[subprocess_id] = sub_case

        # Schedule the start of the subprocess
        first_activity: str = sub_case.template.start_activities[0]
        event = Event(
            case_id=subprocess_id,
            activity=first_activity,
            event_type=EventType.START,
            timestamp=start_time + timedelta(seconds=10),
            process_id=sub_template.name,
            parent_case_id=parent_case.case_id,
            attributes={'fractal_depth': depth + 1}
        )
        self._schedule_event(event)

    # --- Setup and Utility Methods ---

    def _generate_random_process_templates(self) -> Dict[str, ProcessTemplate]:
        """Generates a set of random, structurally different process templates."""
        templates: Dict[str, ProcessTemplate] = {}
        configs = [
            {'name': 'Simple_Process', 'count': (4, 7)},
            {'name': 'Medium_Process', 'count': (8, 12)},
            {'name': 'Complex_Process', 'count': (13, 18)},
            {'name': 'Micro_Process', 'count': (2, 4)},
        ]
        for config in configs:
            num_activities: int = random.randint(*config['count'])
            activities: List[str] = self._generate_random_activity_names(num_activities)
            dependencies: Dict[str, List[str]] = self._generate_random_dependencies(activities)
            decision_points: Dict[str, List[str]] = self._generate_random_decision_points(activities, dependencies)
            
            # Add decision outcomes to the activity list for the template
            all_activities = activities.copy()
            for outcomes in decision_points.values():
                all_activities.extend(outcomes)
                # Ensure decision outcomes have dependencies on the decision point
                for decision, outs in decision_points.items():
                    for out in outs:
                        if out not in dependencies:
                            dependencies[out] = [decision]

            templates[config['name']] = ProcessTemplate(
                name=config['name'],
                activities=list(set(all_activities)),
                dependencies=dependencies,
                decision_points=decision_points
            )
        return templates

    def _generate_random_activity_names(self, count: int) -> List[str]:
        """Generates a list of unique, random activity names."""
        used_names: Set[str] = set()
        activities: List[str] = []
        for i in range(count):
            while True:
                name = f"{random.choice(ACTIVITY_PREFIXES)}_{random.choice(ACTIVITY_SUFFIXES)}_{i+1:02d}"
                if name not in used_names:
                    used_names.add(name)
                    activities.append(name)
                    break
        return activities

    def _generate_random_dependencies(self, activities: List[str]) -> Dict[str, List[str]]:
        """Generates a random but logical dependency graph."""
        dependencies: Dict[str, List[str]] = {}
        for i, activity in enumerate(activities):
            if i == 0:
                continue
            possible_deps = activities[:i]
            dep_count = random.randint(1, min(2, len(possible_deps)))
            dependencies[activity] = random.sample(possible_deps, dep_count)
        return dependencies

    def _generate_random_decision_points(self, activities: List[str], dependencies: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Generates random decision points in the process."""
        decision_points: Dict[str, List[str]] = {}
        # Make non-start activities potential decision points
        possible_decision_nodes = [act for act in activities if dependencies.get(act)]
        if not possible_decision_nodes:
            return {}

        num_decisions = max(1, len(activities) // 5)
        decision_activities = random.sample(possible_decision_nodes, min(num_decisions, len(possible_decision_nodes)))
        
        for i, activity in enumerate(decision_activities):
            outcomes = [f"{activity}_Option_{chr(65+j)}" for j in range(random.randint(2, 3))]
            decision_points[activity] = outcomes
        return decision_points

    def save_to_csv(self, events: List[Event], filename: str) -> None:
        """Saves a list of events to a CSV file."""
        if not events:
            return
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['case:concept:name', 'concept:name', 'livecycle:type', 'time:timestamp', 'process_id', 'parent_case_id', 'correlation_id', 'attributes']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for event in events:
                writer.writerow({
                    'case:concept:name': event.case_id,
                    'concept:name': event.activity,
                    'livecycle:type': event.event_type.value,
                    'time:timestamp': event.timestamp.isoformat(),
                    'process_id': event.process_id,
                    'parent_case_id': event.parent_case_id or '',
                    'correlation_id': event.correlation_id,
                    'attributes': str(event.attributes)
                })

# --- Demonstration ---

def demonstrate_configurable_generator():
    """Demonstrates the generator with different configurations and analyzes the output."""
    print("Refactored Event Stream Generator Demo")
    print("=" * 50)
    
    configs = [
        # {
        #     'name': 'High_Concurrency_Low_OOO',
        #     'concurrency_percentage': 0.9,
        #     'out_of_order_strength': 0.1,
        #     'fractal_behavior_strength': 0.1,
        #     "long_term_dependency_strength": 0.8,
        #     "non_linear_dependency_strength": 0.8,
        #     "temporal_dependency_strength": 0.8,
        # },
        {
            'name': 'Downstream_Task_Dataset',
            'concurrency_percentage': 0.1,
            'out_of_order_strength': 0.9,
            'fractal_behavior_strength': 0.9,
            "long_term_dependency_strength": 0.9,
            "non_linear_dependency_strength": 0.9,
            "temporal_dependency_strength": 0.9,
        }
    ]
    
    for config in configs:
        print(f"\n--- Running Configuration: {config['name']} ---")
        generator = EventStreamGenerator(
            concurrency_percentage=config['concurrency_percentage'],
            out_of_order_strength=config['out_of_order_strength'],
            fractal_behavior_strength=config['fractal_behavior_strength'],
            long_term_dependency_strength=config['long_term_dependency_strength'],
            non_linear_dependency_strength=config['non_linear_dependency_strength'],
            temporal_dependency_strength=config['temporal_dependency_strength'],
            seed=42
        )
        
        events_list: List[Event] = []
        print("Generating events...")
        for i, event in enumerate(generator.generate_event_stream(max_events=1000)):
            events_list.append(event)
            if (i + 1) % 100 == 0:
                print(f"  ... {i+1} events generated")

        print(f"  Total events generated: {len(events_list)}")
        
        filename = f"result_{config['name'].lower()}.csv"
        generator.save_to_csv(events_list, filename)
        print(f"  Saved to: {filename}")

        # Analysis
        ooo_count = sum(1 for e in events_list if e.attributes.get('out_of_order', False))
        fractal_count = sum(1 for e in events_list if 'fractal_depth' in e.attributes)
        
        # Check for true out-of-order delivery by comparing timestamp order to delivery order
        true_ooo_violations = 0
        sorted_by_timestamp = sorted(events_list, key=lambda e: e.timestamp)
        for i in range(len(events_list)):
            if events_list[i] != sorted_by_timestamp[i]:
                true_ooo_violations += 1

        print(f"  Out-of-order attribute set: {ooo_count} ({ooo_count/len(events_list)*100:.1f}%)")
        print(f"  True out-of-order violations: {true_ooo_violations} ({true_ooo_violations/len(events_list)*100:.1f}%)")
        print(f"  Fractal events generated: {fractal_count}")

if __name__ == "__main__":
    demonstrate_configurable_generator()
