import random
import time
import csv
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Generator, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import heapq
import numpy as np
from collections import defaultdict, deque
import uuid

class EventType(Enum):
    START = "start"
    COMPLETE = "complete"
    MILESTONE = "milestone"
    DECISION = "decision"
    ERROR = "error"
    RETRY = "retry"

@dataclass
class Event:
    """Represents a single event in the stream"""
    case_id: str
    activity: str
    event_type: EventType
    timestamp: datetime
    attributes: Dict[str, Any] = field(default_factory=dict)
    process_id: str = ""
    parent_case_id: Optional[str] = None
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __lt__(self, other):
        return self.timestamp < other.timestamp

@dataclass
class ScheduledEvent:
    """Scheduled event with execution time"""
    event: Event
    execution_time: datetime
    dependencies: Set[str] = field(default_factory=set)
    
    def __lt__(self, other):
        return self.execution_time < other.execution_time

class ProcessTemplate:
    """Randomly generated process workflow"""
    
    def __init__(self, name: str, activities: List[str], 
                 dependencies: Dict[str, List[str]] = None,
                 decision_points: Dict[str, List[str]] = None,
                 parallel_groups: Dict[str, List[str]] = None):
        self.name = name
        self.activities = activities
        self.dependencies = dependencies or {}
        self.decision_points = decision_points or {}
        self.parallel_groups = parallel_groups or {}

class ActivityInstance:
    """Represents a running activity instance"""
    
    def __init__(self, activity: str, case_id: str, start_time: datetime):
        self.activity = activity
        self.case_id = case_id
        self.start_time = start_time
        self.expected_duration = random.uniform(0.5, 10.0)  # minutes
        self.is_concurrent = False

class Case:
    """Represents a single case with sophisticated state tracking"""
    
    def __init__(self, case_id: str, process_template: ProcessTemplate):
        self.case_id = case_id
        self.template = process_template
        self.completed_activities = set()
        self.started_activities = set()
        self.running_activities = {}  # activity -> ActivityInstance
        self.pending_activities = set(process_template.activities)
        self.attributes = {}
        self.start_time = datetime.now()
        self.long_term_memory = deque(maxlen=100)
        self.decision_history = []
        
    def can_start_activity(self, activity: str, allow_concurrent: bool) -> bool:
        """Check if activity can start based on dependencies and concurrency"""
        if activity in self.started_activities:
            return False
            
        # Check dependencies
        deps = self.template.dependencies.get(activity, [])
        if not all(dep in self.completed_activities for dep in deps):
            return False
            
        # Check concurrency constraints
        if not allow_concurrent and self.running_activities:
            return False
            
        return True
    
    def get_available_activities(self, allow_concurrent: bool) -> List[str]:
        """Get activities that can be started now"""
        available = []
        for activity in self.pending_activities:
            if self.can_start_activity(activity, allow_concurrent):
                available.append(activity)
        return available

class SophisticatedScheduler:
    """Advanced scheduler managing concurrency and dependencies"""
    
    def __init__(self, concurrency_percentage: float = 0.5):
        self.concurrency_percentage = concurrency_percentage
        self.event_queue = []  # Priority queue of ScheduledEvent
        self.active_cases = {}
        self.global_running_activities = {}
        self.case_start_times = {}
        
    def should_allow_concurrency(self) -> bool:
        """Determine if next activity should be concurrent"""
        return random.random() < self.concurrency_percentage
    
    def schedule_event(self, event: Event, execution_time: datetime, dependencies: Set[str] = None):
        """Schedule an event for future execution"""
        scheduled = ScheduledEvent(event, execution_time, dependencies or set())
        heapq.heappush(self.event_queue, scheduled)
    
    def get_next_scheduled_event(self, current_time: datetime) -> Optional[Event]:
        """Get the next event that should be executed"""
        while self.event_queue:
            scheduled = heapq.heappop(self.event_queue)
            if scheduled.execution_time <= current_time:
                # Check if dependencies are satisfied
                if all(dep in self.global_running_activities for dep in scheduled.dependencies):
                    return scheduled.event
                else:
                    # Reschedule for later
                    scheduled.execution_time = current_time + timedelta(minutes=1)
                    heapq.heappush(self.event_queue, scheduled)
                    continue
            else:
                # Put it back and wait
                heapq.heappush(self.event_queue, scheduled)
                break
        return None

class EventStreamGenerator:
    """Sophisticated event stream generator with configurable features"""
    
    def __init__(self, 
                 temporal_dependency_strength: float = 1.0,
                 long_term_dependency_strength: float = 1.0,
                 non_linear_dependency_strength: float = 1.0,
                 out_of_order_strength: float = 1.0,
                 fractal_behavior_strength: float = 1.0,
                 concurrency_percentage: float = 0.5,
                 seed: int = None):
        
        if seed:
            random.seed(seed)
            np.random.seed(seed)
        
        # Feature strength parameters (0-1)
        self.temporal_dependency_strength = max(0, min(1, temporal_dependency_strength))
        self.long_term_dependency_strength = max(0, min(1, long_term_dependency_strength))
        self.non_linear_dependency_strength = max(0, min(1, non_linear_dependency_strength))
        self.out_of_order_strength = max(0, min(1, out_of_order_strength))
        self.fractal_behavior_strength = max(0, min(1, fractal_behavior_strength))
        self.concurrency_percentage = max(0, min(1, concurrency_percentage))
        
        # Core components
        self.scheduler = SophisticatedScheduler(concurrency_percentage)
        self.global_memory = deque(maxlen=1000)
        self.pattern_history = defaultdict(list)
        self.current_time = datetime.now()
        self.case_counter = 0
        
        # Generate random process templates
        self.templates = self._generate_random_process_templates()
        
        # Configuration
        self.max_concurrent_cases = 20
        self.max_delay_minutes = 30
        
    def _generate_random_activity_names(self, count: int) -> List[str]:
        """Generate random activity names"""
        prefixes = ['Process', 'Review', 'Analyze', 'Check', 'Validate', 'Submit', 
                   'Approve', 'Execute', 'Monitor', 'Report', 'Update', 'Create',
                   'Verify', 'Send', 'Receive', 'Transform', 'Calculate']
        suffixes = ['Data', 'Request', 'Document', 'Form', 'Report', 'Status',
                   'Record', 'File', 'Item', 'Task', 'Order', 'Invoice',
                   'Profile', 'Account', 'Transaction', 'Message']
        
        activities = []
        used_names = set()
        
        for i in range(count):
            while True:
                name = f"{random.choice(prefixes)}_{random.choice(suffixes)}_{i+1:02d}"
                if name not in used_names:
                    used_names.add(name)
                    activities.append(name)
                    break
        
        return activities
    
    def _generate_random_dependencies(self, activities: List[str]) -> Dict[str, List[str]]:
        """Generate random but logical dependencies"""
        dependencies = {}
        
        for i, activity in enumerate(activities):
            if i == 0:
                continue  # First activity has no dependencies
            
            # Each activity depends on 0-2 previous activities
            possible_deps = activities[:i]
            dep_count = random.randint(0, min(2, len(possible_deps)))
            
            if dep_count > 0:
                deps = random.sample(possible_deps, dep_count)
                dependencies[activity] = deps
        
        return dependencies
    
    def _generate_random_decision_points(self, activities: List[str]) -> Dict[str, List[str]]:
        """Generate random decision points"""
        decision_points = {}
        
        # 20% of activities are decision points
        decision_activities = random.sample(activities, max(1, len(activities) // 5))
        
        for activity in decision_activities:
            # Each decision point has 2-3 possible outcomes
            outcome_count = random.randint(2, 3)
            outcomes = []
            
            for i in range(outcome_count):
                outcome_name = f"{activity}_Option_{chr(65+i)}"  # A, B, C...
                outcomes.append(outcome_name)
            
            decision_points[activity] = outcomes
        
        return decision_points
    
    def _generate_random_process_templates(self) -> Dict[str, ProcessTemplate]:
        """Generate multiple random process templates"""
        templates = {}
        
        template_configs = [
            {'name': 'Simple_Process', 'activity_count': random.randint(4, 7)},
            {'name': 'Medium_Process', 'activity_count': random.randint(8, 12)},
            {'name': 'Complex_Process', 'activity_count': random.randint(13, 18)},
            {'name': 'Micro_Process', 'activity_count': random.randint(2, 4)},
        ]
        
        for config in template_configs:
            activities = self._generate_random_activity_names(config['activity_count'])
            dependencies = self._generate_random_dependencies(activities)
            decision_points = self._generate_random_decision_points(activities)
            
            # Add decision outcomes to activities list
            for outcomes in decision_points.values():
                activities.extend(outcomes)
            
            templates[config['name']] = ProcessTemplate(
                name=config['name'],
                activities=activities,
                dependencies=dependencies,
                decision_points=decision_points
            )
        
        return templates
    
    def _apply_temporal_influence(self, case: Case, activity: str) -> float:
        """Apply temporal dependency influence with configurable strength"""
        base_delay = random.uniform(0.5, 5.0)
        
        if self.temporal_dependency_strength == 0:
            return base_delay
        
        # Recent activity influence
        if case.long_term_memory:
            memory_list = list(case.long_term_memory)
            recent_pattern = memory_list[-3:] if len(memory_list) >= 3 else memory_list
            pattern_complexity = len(set(evt.activity for evt in recent_pattern))
            
            temporal_factor = 1 + (pattern_complexity * 0.3 * self.temporal_dependency_strength)
            base_delay *= temporal_factor
        
        return base_delay
    
    def _apply_long_term_influence(self, case: Case) -> float:
        """Apply long-term dependency influence"""
        if self.long_term_dependency_strength == 0:
            return 1.0
        
        if len(case.completed_activities) > 3:
            long_term_factor = 1 + (len(case.completed_activities) * 0.15 * self.long_term_dependency_strength)
            return min(long_term_factor, 3.0)  # Cap the influence
        
        return 1.0
    
    def _apply_non_linear_decision(self, case: Case, activity: str) -> Tuple[str, Dict[str, Any]]:
        """Apply non-linear decision logic with configurable strength"""
        attributes = {}
        
        if activity in case.template.decision_points and self.non_linear_dependency_strength > 0:
            options = case.template.decision_points[activity]
            
            # Non-linear factors
            history_factor = (len(case.completed_activities) % 5) / 4.0
            global_factor = (len(self.global_memory) % 7) / 6.0
            random_factor = random.random()
            
            # Complex decision with configurable strength
            decision_score = (
                history_factor * 0.3 * self.non_linear_dependency_strength +
                global_factor * 0.2 * self.non_linear_dependency_strength +
                random_factor * (1 - self.non_linear_dependency_strength * 0.5) +
                np.sin(len(case.completed_activities)) * 0.1 * self.non_linear_dependency_strength
            )
            
            option_index = int(decision_score * len(options)) % len(options)
            decision_outcome = options[option_index]
            
            attributes.update({
                'decision_score': decision_score,
                'decision_outcome': decision_outcome,
                'non_linear_strength': self.non_linear_dependency_strength
            })
            
            # Add decision outcome to case
            case.pending_activities.add(decision_outcome)
            case.template.activities.append(decision_outcome)
        
        return activity, attributes
    
    def _apply_out_of_order_delivery(self, event: Event) -> Event:
        """Apply out-of-order delivery with configurable strength"""
        if self.out_of_order_strength > 0 and random.random() < (0.2 * self.out_of_order_strength):
            max_delay = self.max_delay_minutes * self.out_of_order_strength
            delay_minutes = random.uniform(-max_delay, max_delay)
            event.timestamp += timedelta(minutes=delay_minutes)
            event.attributes.update({
                'delivery_delay': delay_minutes,
                'out_of_order': True,
                'out_of_order_strength': self.out_of_order_strength
            })
        
        return event
    
    def _generate_fractal_subprocess(self, parent_case: Case, depth: int = 0) -> List[Event]:
        """Generate fractal subprocess with configurable strength"""
        if (self.fractal_behavior_strength == 0 or 
            depth >= int(3 * self.fractal_behavior_strength) or
            len(parent_case.completed_activities) < 2):
            return []
        
        events = []
        subprocess_id = f"{parent_case.case_id}_fractal_{depth}"
        
        # Create scaled-down version of parent activities
        completed_list = list(parent_case.completed_activities)
        fractal_activities = completed_list[:max(1, int(3 * self.fractal_behavior_strength))]
        
        for i, activity in enumerate(fractal_activities):
            scaled_activity = f"Fractal_{activity}_{depth}"
            base_time = self.current_time + timedelta(minutes=i * 0.5)
            
            # Start event
            start_event = Event(
                case_id=subprocess_id,
                activity=scaled_activity,
                event_type=EventType.START,
                timestamp=base_time,
                process_id=f"fractal_depth_{depth}",
                parent_case_id=parent_case.case_id,
                attributes={
                    'fractal_depth': depth,
                    'scale_factor': self.fractal_behavior_strength * (0.7 ** depth),
                    'parent_activity': activity
                }
            )
            
            # Complete event
            complete_event = Event(
                case_id=subprocess_id,
                activity=scaled_activity,
                event_type=EventType.COMPLETE,
                timestamp=base_time + timedelta(minutes=0.5),
                process_id=f"fractal_depth_{depth}",
                parent_case_id=parent_case.case_id,
                attributes={
                    'fractal_depth': depth,
                    'scale_factor': self.fractal_behavior_strength * (0.7 ** depth),
                    'parent_activity': activity
                }
            )
            
            events.extend([start_event, complete_event])
        
        return events
    
    def _create_new_case(self) -> Case:
        """Create a new case with random template selection"""
        self.case_counter += 1
        case_id = f"case_{self.case_counter:06d}"
        
        # Select random template
        template_name = random.choice(list(self.templates.keys()))
        template = self.templates[template_name]
        
        case = Case(case_id, template)
        case.attributes.update({
            'creation_time': self.current_time,
            'template': template_name,
            'global_case_count': self.case_counter
        })
        
        return case
    
    def generate_event_stream(self, max_events: int = 1000) -> Generator[Event, None, None]:
        """Main generator with sophisticated scheduling and proper concurrency control"""
        events_generated = 0
        all_events = []
        loop_counter = 0
        max_loops = 1000  # Safety limit
        
        # Initialize with first case
        initial_case = self._create_new_case()
        self.scheduler.active_cases[initial_case.case_id] = initial_case
        
        # Process all cases to completion first, then sort by timestamp
        while loop_counter < max_loops:
            loop_counter += 1
            
            # Create new cases periodically, but ensure we don't create too many
            if (len(self.scheduler.active_cases) < self.max_concurrent_cases and 
                len(all_events) < max_events * 0.8 and
                random.random() < 0.15):
                new_case = self._create_new_case()
                self.scheduler.active_cases[new_case.case_id] = new_case
            
            # If no active cases, create one more to continue
            if not self.scheduler.active_cases and len(all_events) < max_events * 0.9:
                new_case = self._create_new_case()
                self.scheduler.active_cases[new_case.case_id] = new_case
            
            # Process active cases
            cases_to_process = list(self.scheduler.active_cases.items())
            if not cases_to_process:
                break  # No more cases to process
            
            any_case_progressed = False
            
            for case_id, case in cases_to_process:
                case_events = self._process_case_activities(case)
                if case_events:
                    all_events.extend(case_events)
                    any_case_progressed = True
                
                # Check if case is complete
                if (not case.running_activities and 
                    not case.get_available_activities(True) and
                    len(case.started_activities) > 0):  # Ensure case actually did something
                    del self.scheduler.active_cases[case_id]
            
            # If no cases progressed and we have enough events, break
            if not any_case_progressed and len(all_events) > 20:
                break
            
            # Safety check - if we have enough events, stop creating new cases
            if len(all_events) >= max_events:
                break
        
        print(f"Debug: Generated {len(all_events)} events in {loop_counter} loops")
        
        # Sort all events by timestamp and yield them
        all_events.sort(key=lambda e: e.timestamp)
        
        for event in all_events[:max_events]:
            self.global_memory.append(event)
            yield event
    
    def _process_case_activities(self, case: Case) -> List[Event]:
        """Process all activities for a single case with proper concurrency control"""
        case_events = []
        current_case_time = case.start_time
        
        # Track the last completion time for sequential activities
        last_completion_time = current_case_time
        
        # Safety counters to prevent infinite loops
        activity_counter = 0
        max_activities_per_case = 25
        loop_counter = 0
        max_loops = 100
        
        while activity_counter < max_activities_per_case and loop_counter < max_loops:
            loop_counter += 1
            
            # Get available activities
            allow_concurrent = self.scheduler.should_allow_concurrency()
            available_activities = case.get_available_activities(allow_concurrent)
            
            if not available_activities:
                # Check if we have decision points that haven't been processed
                decision_made = False
                decision_check_count = 0
                
                for completed_activity in list(case.completed_activities):
                    decision_check_count += 1
                    if decision_check_count > 10:  # Prevent too many decision checks
                        break
                        
                    if completed_activity in case.template.decision_points:
                        for decision_outcome in case.template.decision_points[completed_activity]:
                            if (decision_outcome not in case.completed_activities and 
                                decision_outcome not in case.started_activities and
                                decision_outcome not in case.pending_activities):
                                
                                # Only add if dependencies are met
                                deps = case.template.dependencies.get(decision_outcome, [])
                                if all(dep in case.completed_activities for dep in deps):
                                    case.pending_activities.add(decision_outcome)
                                    decision_made = True
                                    break
                    
                    if decision_made:
                        break
                
                if not decision_made:
                    break  # No more activities available
                else:
                    continue  # Try again with new decision outcomes
            
            # Process next activity
            activity = available_activities[0]  # Take first available
            activity_counter += 1
            
            # Apply temporal and long-term influences
            temporal_delay = self._apply_temporal_influence(case, activity)
            long_term_factor = self._apply_long_term_influence(case)
            total_delay = temporal_delay * long_term_factor
            
            # Determine start time based on concurrency
            if allow_concurrent or not case.running_activities:
                # Concurrent: can start with some delay from current time
                start_time = current_case_time + timedelta(minutes=total_delay)
            else:
                # Sequential: must wait for last activity to complete
                start_time = max(
                    last_completion_time + timedelta(minutes=total_delay),
                    current_case_time + timedelta(minutes=total_delay)
                )
            
            # Apply decision logic
            actual_activity, decision_attrs = self._apply_non_linear_decision(case, activity)
            
            # Create activity instance
            duration = random.uniform(0.5, 8.0)
            instance = ActivityInstance(actual_activity, case.case_id, start_time)
            instance.expected_duration = duration
            instance.is_concurrent = allow_concurrent
            
            # Calculate completion time
            completion_time = start_time + timedelta(minutes=duration)
            
            # Create start event
            start_event = Event(
                case_id=case.case_id,
                activity=actual_activity,
                event_type=EventType.START,
                timestamp=start_time,
                process_id=case.template.name,
                attributes={
                    **decision_attrs,
                    'temporal_delay': temporal_delay,
                    'long_term_factor': long_term_factor,
                    'is_concurrent': allow_concurrent,
                    'expected_duration': duration,
                    'sequence_position': len(case.started_activities)
                }
            )
            
            # Create complete event
            complete_event = Event(
                case_id=case.case_id,
                activity=actual_activity,
                event_type=EventType.COMPLETE,
                timestamp=completion_time,
                process_id=case.template.name,
                attributes={
                    'duration': duration,
                    'was_concurrent': allow_concurrent,
                    'sequence_position': len(case.started_activities)
                }
            )
            
            # Apply out-of-order delivery
            start_event = self._apply_out_of_order_delivery(start_event)
            complete_event = self._apply_out_of_order_delivery(complete_event)
            
            case_events.extend([start_event, complete_event])
            
            # Update case state
            case.started_activities.add(actual_activity)
            case.completed_activities.add(actual_activity)
            case.pending_activities.discard(actual_activity)
            case.running_activities[actual_activity] = instance
            case.long_term_memory.append(start_event)
            
            # Update timing for next iteration
            if not allow_concurrent:
                last_completion_time = completion_time
            
            current_case_time = start_time + timedelta(minutes=0.1)  # Small increment
            
            # Generate fractal subprocess (reduce frequency to avoid too many events)
            if random.random() < 0.05 * self.fractal_behavior_strength:
                fractal_events = self._generate_fractal_subprocess(case)
                case_events.extend(fractal_events[:4])  # Limit fractal events
        
        return case_events
    
    def save_to_csv(self, events: List[Event], filename: str = "result.csv"):
        """Save events to CSV file"""
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'case_id', 'activity', 'event_type', 'timestamp', 'process_id',
                'parent_case_id', 'correlation_id', 'attributes'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for event in events:
                writer.writerow({
                    'case_id': event.case_id,
                    'activity': event.activity,
                    'event_type': event.event_type.value,
                    'timestamp': event.timestamp.isoformat(),
                    'process_id': event.process_id,
                    'parent_case_id': event.parent_case_id or '',
                    'correlation_id': event.correlation_id,
                    'attributes': str(event.attributes) if event.attributes else ''
                })

def demonstrate_configurable_generator():
    """Demonstrate the configurable event stream generator"""
    print("Configurable Event Stream Generator Demo")
    print("=" * 50)
    
    # Test different configurations
    configs = [
        {
            'name': 'High Concurrency',
            'temporal_dependency_strength': 0.8,
            'long_term_dependency_strength': 0.7,
            'non_linear_dependency_strength': 0.9,
            'out_of_order_strength': 0.6,
            'fractal_behavior_strength': 0.5,
            'concurrency_percentage': 0.8  # High concurrency
        },
        {
            'name': 'Low Concurrency',
            'temporal_dependency_strength': 0.8,
            'long_term_dependency_strength': 0.7,
            'non_linear_dependency_strength': 0.9,
            'out_of_order_strength': 0.6,
            'fractal_behavior_strength': 0.5,
            'concurrency_percentage': 0.2  # Low concurrency - should see sequential patterns
        }
    ]
    
    for config in configs:
        print(f"\n--- {config['name']} Configuration ---")
        print(f"Concurrency Percentage: {config['concurrency_percentage']*100:.0f}%")
        
        generator = EventStreamGenerator(
            temporal_dependency_strength=config['temporal_dependency_strength'],
            long_term_dependency_strength=config['long_term_dependency_strength'],
            non_linear_dependency_strength=config['non_linear_dependency_strength'],
            out_of_order_strength=config['out_of_order_strength'],
            fractal_behavior_strength=config['fractal_behavior_strength'],
            concurrency_percentage=config['concurrency_percentage'],
            # seed=42
        )
        
        events = []
        case_counts = defaultdict(int)
        
        print("Generating events...")
        event_count = 0
        for i, event in enumerate(generator.generate_event_stream(max_events=5000)):
            events.append(event)
            case_counts[event.case_id] += 1
            event_count += 1
            
            # Progress indicator
            if event_count % 10 == 0:
                print(f"  Progress: {event_count} events generated...")
            
            if i < 12:
                timestamp_str = event.timestamp.strftime('%H:%M:%S.%f')[:-3]
                concurrent_marker = " [C]" if event.attributes.get('is_concurrent', False) else " [S]"
                print(f"  {i+1:2d}: {event.case_id} | {event.activity[:18]:18s} | {event.event_type.value:8s} | {timestamp_str}{concurrent_marker}")
            elif i == 12:
                print("     ... (remaining events truncated)")
        
        print(f"  Completed generation of {len(events)} events")
        
        # Save to CSV
        filename = f"result_{config['name'].lower().replace(' ', '_')}.csv"
        generator.save_to_csv(events, filename)
        
        # Analysis
        out_of_order = sum(1 for e in events if e.attributes.get('out_of_order', False))
        concurrent_starts = sum(1 for e in events if e.event_type == EventType.START and e.attributes.get('is_concurrent', False))
        sequential_starts = sum(1 for e in events if e.event_type == EventType.START and not e.attributes.get('is_concurrent', False))
        fractal = sum(1 for e in events if 'fractal_depth' in e.attributes)
        
        # Analyze concurrency patterns per case
        case_patterns = defaultdict(list)
        for event in events:
            if event.event_type == EventType.START:
                case_patterns[event.case_id].append({
                    'activity': event.activity,
                    'timestamp': event.timestamp,
                    'concurrent': event.attributes.get('is_concurrent', False)
                })
        
        # Check for sequential patterns (start immediately after complete)
        sequential_pairs = 0
        total_pairs = 0
        for case_id, case_events in case_patterns.items():
            if len(case_events) > 1:
                for i in range(len(case_events) - 1):
                    total_pairs += 1
                    curr_event = case_events[i]
                    next_event = case_events[i + 1]
                    if not curr_event['concurrent']:
                        # For sequential events, check if next starts close to current completion
                        # (This is approximate since we don't track exact completion times here)
                        sequential_pairs += 1
        
        print(f"  Generated: {len(events)} events across {len(case_counts)} cases")
        print(f"  Out-of-order: {out_of_order} ({out_of_order/len(events)*100:.1f}%)")
        print(f"  Concurrent starts: {concurrent_starts} ({concurrent_starts/(concurrent_starts+sequential_starts)*100:.1f}%)")
        print(f"  Sequential starts: {sequential_starts} ({sequential_starts/(concurrent_starts+sequential_starts)*100:.1f}%)")
        print(f"  Fractal events: {fractal}")
        print(f"  Expected concurrency: {config['concurrency_percentage']*100:.0f}%")
        print(f"  Saved to: {filename}")
    
    return events

if __name__ == "__main__":
    events = demonstrate_configurable_generator()