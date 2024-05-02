# https://realpython.com/python-interface/

# an interface acts as a blueprint for designing classes.
# interfaces defines methods which are abstract in nature and the class which inherits the interface must implement them.


# informal interface
# An informal Python interface is a class that defines methods that can be overridden, but there’s no strict enforcement.

# eg.
class InformalParserInterface:
    def load_data_source(self, path: str, file_name: str) -> str:
        """Load in the data set"""
        pass

    def extract_text(self, full_file_name: str) -> dict:
        """Extract text from the currently loaded data set"""
        pass

    # methods in this class are defined but not implemented.


# implementing concrete class from informal interface
# pdfparser class implements the informal interface
class PdfParserNew(InformalParserInterface):
    """Extract text from a PDF."""

    def load_data_source(self, path: str, file_name: str) -> str:
        """Overrides InformalParserInterface.load_data_source()"""
        pass

    def extract_text(self, full_file_name: str) -> dict:
        """Overrides InformalParserInterface.extract_text()"""
        pass

# exmlparser class implements the informal interface
class EmlParserNew(InformalParserInterface):
    """Extract text from an email."""

    def load_data_source(self, path: str, file_name: str) -> str:
        """Overrides InformalParserInterface.load_data_source()"""
        pass

    def extract_text_from_email(self, full_file_name: str) -> dict:
        """A method defined only in EmlParser.
        Does not override InformalParserInterface.extract_text()
        """
        pass








