import multiprocessing
import os
import sys
from collections import defaultdict
from typing import Dict, Tuple, Any, List

from distributed_event_factory.core.abstract_datasource import DataSource
from distributed_event_factory.core.datasource import GenericDataSource
from distributed_event_factory.core.datasource_id import DataSourceId
from distributed_event_factory.core.start_datasource import StartDataSource
from distributed_event_factory.event_factory import EventFactory
from distributed_event_factory.provider.activity.activity_provider import ConstantActivityProvider
from distributed_event_factory.provider.event.event_provider import EventDataProvider, CustomEventDataProvider
from distributed_event_factory.provider.eventselection.generic_probability_event_selection_provider import \
    GenericProbabilityEventSelectionProvider
from distributed_event_factory.provider.sink.test.test_sink_parser import TestSinkParser
from distributed_event_factory.provider.transition.duration.constant_duration import ConstantDurationProvider
from distributed_event_factory.provider.transition.transition.constant_transition import ConstantTransitionProvider
from gedi_streams.def_configurations.sink.gedi_sink import GEDIAdapter

mk = {'a': {'b': 0.116, 'd': 0.8530000000000001, 'j': 0.008999999999999996, 'e': 0.012999999999999996, 'h': 0.008999999999999996}, 'b': {'e': 0.08090452261306977, 'n': 0.07788944723618518, 'k': 0.1733668341708638, 's': 0.06683417085427502, 'c': 0.15376884422111398, 'a': 0.05175879396985209, 'p': 0.034673366834172756, 'm': 0.03216080402010227, 'q': 0.030150753768845878, 'r': 0.026633165829147192, 'd': 0.04874371859296751, 'i': 0.03165829145728817, 'h': 0.08190954773869796, 'f': 0.02763819095477539, 'l': 0.034673366834172756, 'o': 0.02713567839196129, 'g': 0.01206030150753835, 't': 0.008040201005025565}, 'e': {'d': 0.03356890459363957, 'c': 0.14664310954063606, 's': 0.04770318021201413, 'l': 0.03003533568904593, 't': 0.005300353356890455, 'q': 0.03533568904593639, 'j': 0.2968197879858657, 'k': 0.127208480565371, 'o': 0.02473498233215547, 'a': 0.03003533568904593, 'n': 0.06537102473498234, 'f': 0.02473498233215547, 'i': 0.03003533568904593, 'm': 0.03180212014134275, 'p': 0.03180212014134275, 'r': 0.03533568904593639, 'g': 0.0035335689045936374}, 'd': {'j': 0.006999999999999996, 'n': 0.8620000000000001, 'h': 0.014999999999999996, 'b': 0.095, 'e': 0.020999999999999998}, 'j': {'n': 0.03356890459363957, 'c': 0.1590106007067138, 'p': 0.04593639575971731, 'r': 0.04593639575971731, 'q': 0.03180212014134275, 'b': 0.25265017667844525, 's': 0.07420494699646643, 'm': 0.02473498233215547, 'l': 0.02296819787985865, 'd': 0.02650176678445229, 'f': 0.03180212014134275, 'k': 0.15371024734982333, 'a': 0.01590106007067137, 'i': 0.04063604240282685, 'o': 0.02826855123674911, 't': 0.007067137809187275, 'g': 0.005300353356890455}, 'n': {'f': 0.4468937875751749, 'k': 0.42985971943890144, 'b': 0.07114228456914219, 'j': 0.014529058116233264, 'e': 0.020541082164329788, 'h': 0.01703406813627348}, 'f': {'o': 0.8975903614457831, 'e': 0.014056224899598388, 'b': 0.05823293172690763, 'j': 0.01706827309236947, 'h': 0.01305220883534136}, 'o': {'b': 0.07028112449799197, 'n': 0.892570281124498, 'j': 0.007028112449799191, 'h': 0.016064257028112445, 'e': 0.014056224899598388}, 'k': {'t': 0.06370575988895952, 'c': 0.5236641221373086, 'g': 0.07078417765439945, 'i': 0.20124913254681506, 'p': 0.050936849410126675, 'e': 0.011103400416376386, 'h': 0.01290770298403755, 'b': 0.054961832061063115, 'j': 0.010687022900762271}, 't': {'k': 0.9420289855072463, 'b': 0.0331262939958592, 'e': 0.012422360248447195, 'h': 0.0041407867494823985, 'j': 0.008281573498964797}, 'c': {'l': 0.24548264384209392, 'k': 0.4538754160721992, 'm': 0.24167855444599384, 'b': 0.03399904897764787, 'e': 0.007370423204944644, 'h': 0.008440323347597898, 'j': 0.009153590109366734}, 'l': {'c': 0.9406350667280287, 'j': 0.00874367234238432, 'b': 0.028071790151865457, 'h': 0.011044638748274932, 'e': 0.011504832029453054}, 'g': {'k': 0.9191176470588235, 'b': 0.04227941176470588, 'h': 0.01286764705882352, 'j': 0.01286764705882352, 'e': 0.01286764705882352}, 'i': {'r': 0.7733812949640417, 'p': 0.16443987667010138, 'j': 0.011305241521069468, 'b': 0.0323741007194262, 'h': 0.009763617677287267, 'e': 0.008735868448099133}, 'r': {'s': 0.7533401849948753, 'p': 0.18396711202467592, 'e': 0.009763617677287267, 'b': 0.03442959917780247, 'h': 0.008221993833505066, 'j': 0.010277492291881333}, 's': {'q': 0.4263370332997133, 'k': 0.3713420787083929, 'p': 0.14228052472251432, 'h': 0.00807265388496535, 'b': 0.03355196770938724, 'j': 0.009838546922301522, 'e': 0.008577194752775685}, 'q': {'s': 0.8379583746283537, 'p': 0.10158572844400963, 'b': 0.03567888999009119, 'j': 0.011397423191279128, 'h': 0.0074331020812689955, 'e': 0.005946481665015196}, 'p': {'s': 0.29239465570402395, 'k': 0.18345323741008185, 'r': 0.16700924974307169, 'i': 0.1916752312435869, 'q': 0.11202466598150655, 'h': 0.0066803699897228655, 'j': 0.008221993833505066, 'e': 0.013360739979445735, 'b': 0.025179856115109273}, 'h': {'b': 0.23801065719360567, 'n': 0.06039076376554174, 'k': 0.18117229129662524, 's': 0.06394316163410302, 'c': 0.15985790408525755, 'l': 0.015985790408525744, 'f': 0.030195381882770864, 'p': 0.03374777975133214, 'a': 0.028419182948490225, 'd': 0.028419182948490225, 'o': 0.0319715808170515, 'i': 0.035523978685612786, 'q': 0.021314387211367664, 'm': 0.030195381882770864, 'g': 0.008880994671403191, 'r': 0.030195381882770864, 't': 0.0017761989342806382}, 'm': {'c': 0.9440820130475331, 'b': 0.02795899347623651, 'e': 0.006523765144455183, 'h': 0.009785647716682777, 'j': 0.011649580615098545}}
project_root = "../"
submodule_path = os.path.join(project_root, "DistributedEventFactory")
sys.path.append(submodule_path)

def init_and_compile_def_using_markov_chain(markov_chain: Dict[str, Dict[str, float]], submodule_path: str) -> EventFactory:

    datasource_definitions: Dict[str, DataSource] = {}

    for state, transitions in markov_chain.items():

        potential_events: list[EventDataProvider] = []

        for next_state, probability in transitions.items():

            custom_event_data_provider: CustomEventDataProvider = CustomEventDataProvider(
                duration_provider=ConstantDurationProvider(1),
                activity_provider=ConstantActivityProvider(state),
                transition_provider=ConstantTransitionProvider(next_state)
            )

            potential_events.append(custom_event_data_provider)

        uniform_event_selection_provider: GenericProbabilityEventSelectionProvider = (
            GenericProbabilityEventSelectionProvider(
                potential_events=potential_events,
                probability_distribution=list(transitions.values())
            )
        )

        generic_datasource: GenericDataSource = GenericDataSource(
            data_source_id=DataSourceId(state),
            group_id="markov_chain",
            event_provider=uniform_event_selection_provider
        )
        datasource_definitions[state] = generic_datasource

    datasource_keys: List[str] = list(datasource_definitions.keys())

    datasource_definitions["<start>"] = GenericDataSource(
        data_source_id=DataSourceId("<start>"),
        group_id="markov_chain",
        event_provider=GenericProbabilityEventSelectionProvider(
                potential_events=[
                    CustomEventDataProvider(
                        duration_provider=ConstantDurationProvider(1),
                        activity_provider=ConstantActivityProvider("<start>"),
                        transition_provider=ConstantTransitionProvider(datasource_keys[0])
                    )
                ],
                probability_distribution=[1.0]
        )
    )

    event_factory: EventFactory = EventFactory()
    for name, generic_datasource in datasource_definitions.items():
        event_factory.add_datasource(name, generic_datasource)

    event_factory.add_sink("gedi-sink", GEDIAdapter("gedi-sink", datasource_keys, multiprocessing.Queue(), disable_console_print=False))

    (event_factory
    # .add_directory(f"{submodule_path}/config/datasource/assemblyline")
     .add_file(f"{submodule_path}/config/simulation/stream.yaml")
     )

    return event_factory


if __name__ == '__main__':
    print("Running")
    init_and_compile_def_using_markov_chain(mk, submodule_path).run()
    print("Done")