from gedi_streams.utils.io_helpers import list_classes_in_file

def test_list_classes_in_file():
    EXPECTED_OUTPUT = ['EventDataFile', 'FeatureExtraction', 'UnsupportedFileExtensionError']
    result = list_classes_in_file("gedi_streams/features/feature_extraction.py")
    assert len(result) == 3
    assert set(result) == set(EXPECTED_OUTPUT)

