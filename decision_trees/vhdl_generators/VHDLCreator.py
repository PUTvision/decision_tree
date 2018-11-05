import abc


class VHDLCreator:
    __metaclass__ = abc.ABCMeta

    def __init__(self, filename: str, entity_name: str, number_of_features: int, number_of_bits_per_feature: int):
        self.current_indent = 0

        self.filename = filename

        self.FILE_EXTENSION = ".vhd"
        self.TESTBENCH_PREFIX = "tb_"
        self.CUSTOM_TYPES_POSTFIX = "_types"

        self._param_entity_name = entity_name
        self._param_testbench_entity_name = self.TESTBENCH_PREFIX + entity_name

        self._filename_custom_types = self.filename + self.CUSTOM_TYPES_POSTFIX + self.FILE_EXTENSION
        self._filename = self.filename + self.FILE_EXTENSION
        self._filename_testbench = self.TESTBENCH_PREFIX + self.filename + self.FILE_EXTENSION
        self._custom_type_name = self.filename + "_t"

        # TODO - maybe this part can be found automatically
        self._number_of_bits_for_class_index = 8
        self._number_of_bits_per_feature = number_of_bits_per_feature
        self._number_of_features = number_of_features

    def _insert_text_line_with_indent(self, text_to_insert) -> str:
        text = ""
        # apply current indent
        text += "\t" * self.current_indent
        text += text_to_insert
        text += "\n"
        return text

    def _add_headers(self) -> str:
        text = ""
        text += self._insert_text_line_with_indent("library IEEE;")
        text += self._insert_text_line_with_indent("use IEEE.STD_LOGIC_1164.ALL;")
        text += self._insert_text_line_with_indent("use IEEE.NUMERIC_STD.ALL;")
        text += self._insert_text_line_with_indent("")

        text += self._add_additional_headers()
        text += self._insert_text_line_with_indent("")

        return text

    @abc.abstractmethod
    def _add_additional_headers(self) -> str:
        return

    def _add_entity(self, output_size_multiplier: int = 1) -> str:
        text = ""

        text += self._insert_text_line_with_indent("entity " + self._param_entity_name + " is")
        text += self._add_entity_generics_section()
        text += self._add_entity_port_section(output_size_multiplier)
        text += self._insert_text_line_with_indent("end " + self._param_entity_name + ";")
        text += self._insert_text_line_with_indent("")

        return text

    @abc.abstractmethod
    def _add_entity_generics_section(self) -> str:
        return

    def _add_entity_port_section(self, output_size_multiplier: int = 1) -> str:
        text = ""

        self.current_indent += 1

        text += self._insert_text_line_with_indent("port (")

        # insert all the ports
        self.current_indent += 1

        text += self._insert_text_line_with_indent("clk" + "\t\t\t\t" + ":" + "\t" + "in std_logic;")
        text += self._insert_text_line_with_indent("rst" + "\t\t\t\t" + ":" + "\t" + "in std_logic;")
        text += self._insert_text_line_with_indent("en" + "\t\t\t\t" + ":" + "\t" + "in std_logic;")

        # input aggregated to one long std_logic_vector
        text += self._insert_text_line_with_indent("input" + "\t\t\t" + ":" + "\t" + "in std_logic_vector("
                                                   + str(self._number_of_features*self._number_of_bits_per_feature)
                                                   + "-1 downto 0);")

        text += self._insert_text_line_with_indent("output" + "\t\t\t" + ":" + "\t" + "out std_logic_vector("
                                                   + str(self._number_of_bits_for_class_index * output_size_multiplier)
                                                   + "-1 downto 0)")

        self.current_indent -= 1

        text += self._insert_text_line_with_indent(");")

        self.current_indent -= 1

        return text

    def _add_architecture(self) -> str:
        text = ""

        text += self._insert_text_line_with_indent("architecture Behavioral of " + self._param_entity_name + " is")
        text += self._insert_text_line_with_indent("")

        self.current_indent += 1

        text += self._add_architecture_component_section()

        text += self._add_architecture_signal_section()

        self.current_indent -= 1

        text += self._insert_text_line_with_indent("begin")
        text += self._insert_text_line_with_indent("")

        self.current_indent += 1

        text += self._add_architecture_process_section()

        self.current_indent -= 1

        text += self._insert_text_line_with_indent("end Behavioral;")

        return text

    @abc.abstractmethod
    def _add_architecture_component_section(self) -> str:
        return

    @abc.abstractmethod
    def _add_architecture_signal_section(self) -> str:
        return

    @abc.abstractmethod
    def _add_architecture_process_section(self) -> str:
        return

    @abc.abstractmethod
    def create_vhdl_file(self, path: str):
        return
