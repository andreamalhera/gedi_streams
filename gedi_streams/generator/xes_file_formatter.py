import re
import xml.etree.ElementTree as ET
from xml.dom import minidom


class XesFileFormatter:
    """Handles XES file formatting operations."""

    @staticmethod
    def remove_extra_lines(elem):
        """Removes extra lines from XML elements."""
        has_words = re.compile("\\w")
        for element in elem.iter():
            if not re.search(has_words, str(element.tail)):
                element.tail = ""
            if not re.search(has_words, str(element.text)):
                element.text = ""

    @staticmethod
    def add_extension_before_traces(xes_file):
        """Adds standard extensions to XES files."""
        # Register the namespace
        ET.register_namespace('', "http://www.xes-standard.org/")

        # Parse the original XML
        tree = ET.parse(xes_file)
        root = tree.getroot()

        # Add extensions
        extensions = [
            {'name': 'Lifecycle', 'prefix': 'lifecycle', 'uri': 'http://www.xes-standard.org/lifecycle.xesext'},
            {'name': 'Time', 'prefix': 'time', 'uri': 'http://www.xes-standard.org/time.xesext'},
            {'name': 'Concept', 'prefix': 'concept', 'uri': 'http://www.xes-standard.org/concept.xesext'}
        ]

        for ext in extensions:
            extension_elem = ET.Element('extension', ext)
            root.insert(0, extension_elem)

        # Add global variables
        globals_config = [
            {
                'scope': 'event',
                'attributes': [
                    {'key': 'lifecycle:transition', 'value': 'complete'},
                    {'key': 'concept:name', 'value': '__INVALID__'},
                    {'key': 'time:timestamp', 'value': '1970-01-01T01:00:00.000+01:00'}
                ]
            },
            {
                'scope': 'trace',
                'attributes': [
                    {'key': 'concept:name', 'value': '__INVALID__'}
                ]
            }
        ]

        for global_var in globals_config:
            global_elem = ET.Element('global', {'scope': global_var['scope']})
            for attr in global_var['attributes']:
                string_elem = ET.SubElement(global_elem, 'string', {'key': attr['key'], 'value': attr['value']})
            root.insert(len(extensions), global_elem)

        # Pretty print the XES
        XesFileFormatter.remove_extra_lines(root)
        xml_str = minidom.parseString(ET.tostring(root)).toprettyxml()
        with open(xes_file, "w") as f:
            f.write(xml_str)