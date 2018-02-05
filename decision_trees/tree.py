import abc

import numpy as np
import sklearn.tree

from decision_trees.utils.convert_to_fixed_point import convert_to_fixed_point


class Split:

    id = 0
    var_idx = 0
    value_to_compare = 0.0

    def __init__(self, id_, var_idx_, value_to_compare_):
        self.id = id_
        self.var_idx = var_idx_
        self.value_to_compare = value_to_compare_

    def show(self):
        print("Split[", self.id, "], var_idx: ", self.var_idx, ", value to compare: ", self.value_to_compare)


class Leaf:

    id = 0
    class_idx = 0

    following_split_IDs = []
    following_split_compare_values = []

    def __init__(self, id_, class_idx_, following_split_IDs_, following_spli_compare_values_):
        self.id = id_
        self.class_idx = class_idx_
        self.following_split_IDs = following_split_IDs_
        self.following_split_compare_values = following_spli_compare_values_

    def show(self):
        print("Leaf[", self.id, "], class_idx: ", self.class_idx)

        print("Following splits IDs: ")
        splits_ids = ""
        for id_ in self.following_split_IDs:
            splits_ids += str(id_) + ", "
        print(splits_ids)

        print("Following split compare values: ")
        compare_values = ""
        for compare_value in self.following_split_compare_values:
            compare_values += str(compare_value) + ", "
        print(compare_values)


# TODO: while the tree structure takes into account the number of bits used for compare values...
# TODO cont: ... it is not implemented in vhdl conversion

# TODO add code that retrains the network on already limited representation

class VHDLcreator:
    __metaclass__ = abc.ABCMeta

    def __init__(self, name: str, number_of_features: int, number_of_bits_per_feature: int):
        self.current_indent = 0

        self.name = name

        self.FILE_EXTENSION = ".vhd"
        self.TESTBENCH_PREFIX = "tb_"
        self.CUSTOM_TYPES_POSTFIX = "_types"

        self._param_entity_name = "" + self.name
        self._param_testbench_entity_name = self.TESTBENCH_PREFIX + self.name

        self._filename_custom_types = self.name + self.CUSTOM_TYPES_POSTFIX + self.FILE_EXTENSION
        self._filename = self.name + self.FILE_EXTENSION
        self._filename_testbench = self.TESTBENCH_PREFIX + self.name + self.FILE_EXTENSION
        self._custom_type_name = self.name + "_t"

        # TODO - maybe this part can be found automatically
        self._number_of_bits_for_class_index = 32
        self._number_of_bits_per_feature = number_of_bits_per_feature
        self._number_of_features = number_of_features

    def _insert_text_line_with_indent(self, text_to_insert):
        text = ""
        # apply current indent
        text += "\t" * self.current_indent
        text += text_to_insert
        text += "\n"
        return text

    def _add_headers(self):
        text = ""
        text += self._insert_text_line_with_indent("library IEEE;")
        text += self._insert_text_line_with_indent("use IEEE.STD_LOGIC_1164.ALL;")
        text += self._insert_text_line_with_indent("use IEEE.NUMERIC_STD.ALL;")
        text += self._insert_text_line_with_indent("")

        text += self._add_additional_headers()
        text += self._insert_text_line_with_indent("")

        return text

    @abc.abstractmethod
    def _add_additional_headers(self):
        return

    def _add_entity(self):
        text = ""

        text += self._insert_text_line_with_indent("entity " + self._param_entity_name + " is")
        text += self._add_entity_generics_section()
        text += self._add_entity_port_section()
        text += self._insert_text_line_with_indent("end " + self._param_entity_name + ";")
        text += self._insert_text_line_with_indent("")

        return text

    @abc.abstractmethod
    def _add_entity_generics_section(self):
        return

    @abc.abstractmethod
    def _add_entity_port_section(self):
        return

    def _add_architecture(self):
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
    def _add_architecture_component_section(self):
        return

    @abc.abstractmethod
    def _add_architecture_signal_section(self):
        return

    @abc.abstractmethod
    def _add_architecture_process_section(self):
        return

    def create_vhdl_file(self):
        # open file for writing
        file_to_write = open(self._filename, "w")
        # add necessary headers
        text = ""
        text += self._add_headers()
        text += self._add_entity()
        text += self._add_architecture()

        file_to_write.write(text)

        file_to_write.close()


class RandomForest(VHDLcreator):

    def __init__(self, number_of_features: int, number_of_bits_per_feature: int):
        self.random_forest = []

        VHDLcreator.__init__(self, "RandomForestTest", number_of_features, number_of_bits_per_feature)

    def build(self, random_forest):
        for i, tree in enumerate(random_forest.estimators_):
            tree_builder = Tree("tree_" + str(i), self._number_of_features, self._number_of_bits_per_feature)
            tree_builder.build(tree)

            self.random_forest.append(tree_builder)

    def predict(self, input_data):
        # first create a dictionary that will store the results
        results = {}
        for tree in self.random_forest:
            # get the result form one of the tree and add it to appropriate element in dict
            tree_result = tree.predict(input_data)
            if tree_result in results:
                results[tree_result] += 1
            else:
                results[tree_result] = 1

        # IMPORTANT - following operations are required to make sure that the result is the same as obtained from scikit
        # the problem (class name, number of votes):
        # 0: 5, 1: 0, 2: 1, 3: 5
        # scikit result - 0 (even though 0 and 3 have the same number of votes)
        # my result - it depends on which value was presented first, so it can be 0 or 3

        # find maximal value
        max_value = max(results.values())
        # and use it to get all pairs that are equal
        max_result = [(key, value) for key, value in results.items() if value == max_value]
        # at the end get element with the lowest key value
        chosen_class = min(max_result, key=lambda t: t[0])[0]

        return chosen_class

    def print_parameters(self):
        for tree in self.random_forest:
            tree.print_parameters()

    def _add_additional_headers(self):
        text = ""
        return text

    def _add_entity_generics_section(self):
        text = ""
        return text

    def _add_entity_port_section(self):
        text = ""
        return text

    def _add_architecture_component_section(self):
        text = ""
        return text

    def _add_architecture_signal_section(self):
        text = ""
        return text

    def _add_architecture_process_section(self):
        text = ""
        return text


class Tree(VHDLcreator):

    def __init__(self, name: str, number_of_features: int, number_of_bits_per_feature: int):
        self._current_split_index = 0
        self._current_leaf_index = 0

        self.splits = []
        self.leaves = []

        VHDLcreator.__init__(self, name, number_of_features, number_of_bits_per_feature)

    def build(self, tree):
        self._current_split_index = 0
        self._current_leaf_index = 0

        self.splits = []
        self.leaves = []

        following_splits_IDs = []
        following_splits_compare_values = []

        features_names = []
        for i in range(0, self._number_of_features):
            features_names.append(i)

        features = [
            features_names[i] if i != sklearn.tree._tree.TREE_UNDEFINED else "undefined!"
            for i in tree.tree_.feature
        ]

        self._preorder(tree.tree_, features, following_splits_IDs, following_splits_compare_values, 0)

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        result_data = np.empty(len(input_data))

        for i in range(len(input_data)):
            result_data[i] = self._predict_one_sample(input_data[i])

        return result_data

    def _predict_one_sample(self, input_data):
        # this code works in a similar way to how vhdl implementation of the tree works

        chosen_class_index = [-1]
        # first calculate all the comparisions
        compare_results = [None] * len(self.splits)
        for i, split in enumerate(self.splits):
            if input_data[split.var_idx] <= split.value_to_compare:
                compare_results[i] = 0
            else:
                compare_results[i] = 1

        # now go through all leaves and check if it following compare values are same as one calculated above
        for leaf in self.leaves:
            number_of_correct_results = 0

            for j in range(0, len(leaf.following_split_IDs)):
                split_id = leaf.following_split_IDs[j]
                expected_result = leaf.following_split_compare_values[j]
                real_result = compare_results[split_id]

                if expected_result == real_result:
                    number_of_correct_results += 1

            if number_of_correct_results == len(leaf.following_split_IDs):
                chosen_class_index = leaf.class_idx

        # find the most important class
        chosen_class = np.argmax(chosen_class_index[0])

        return chosen_class

    def print_parameters(self):
        # self.print_leaves()
        # self.print_splits()
        print("Depth: ", self.find_depth())
        print("Number of splits: ", len(self.splits))
        print("Number of leaves: ", len(self.leaves))

    def print_splits(self):
        print("Splits: ")
        print(len(self.splits))
        for split in self.splits:
            split.show()

    def print_leaves(self):
        print("Leaves: ")
        print(len(self.leaves))
        for leaf in self.leaves:
            leaf.show()

    def find_depth(self):

        following_splits_number = []

        for leaf in self.leaves:
            following_splits_number.append(len(leaf.following_split_IDs))

        return max(following_splits_number)

    def _add_additional_headers(self):
        text = ""
        return text

    def _add_entity_generics_section(self):
        text = ""
        return text

    def _add_entity_port_section(self):
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
                                                   + str(self._number_of_bits_for_class_index)
                                                   + "-1 downto 0)")

        self.current_indent -= 1

        text += self._insert_text_line_with_indent(");")

        self.current_indent -= 1

        return text

    def _add_architecture_component_section(self):
        text = ""
        return text

    def _add_architecture_signal_section(self):
        text = ""

        text += self._insert_text_line_with_indent("type " + "features_t" + "\t" + "is array("
                                                   + str(self._number_of_features) + "-1 downto 0)"
                                                   + " of std_logic_vector(" + str(self._number_of_bits_per_feature)
                                                   + "-1 downto 0);")

        text += self._insert_text_line_with_indent("signal " + "features" + "\t\t:\t" + "features_t"
                                                   + "\t\t\t" + ":= (others=>(others=>'0'));")

        text += self._insert_text_line_with_indent("signal " + "splitResult" + "\t:\t" + "std_logic_vector("
                                                   + str(len(self.splits)) + "-1 downto 0)"
                                                   + "\t\t\t" + ":= (others=>'0');")
        text += self._insert_text_line_with_indent("signal " + "classIndex" + "\t:\t" + "unsigned("
                                                   + str(self._number_of_bits_for_class_index) + "-1 downto 0)"
                                                   + "\t\t\t" + ":= (others=>'0');")

        text += self._insert_text_line_with_indent("")

        return text

    def _add_architecture_process_section(self):
        text = ""

        text += self._add_architecture_input_mapping()
        text += self._add_architecture_process_compare()
        text += self._add_architecture_process_decideClass()
        text += self._insert_text_line_with_indent("output <= std_logic_vector(classIndex);")
        text += self._insert_text_line_with_indent("")

        return text

    def _add_architecture_input_mapping(self):
        text = ""

        for i in range(0, self._number_of_features):
            text += self._insert_text_line_with_indent("features(" + str(i) + ") <= input("
                                                       + str(self._number_of_bits_per_feature*(i+1)-1) + " downto "
                                                       + str(self._number_of_bits_per_feature*i) + ");")

        return text

    def _add_architecture_process_compare(self):
        # create the code for all the compares done in the splits
        text = ""

        text += self._insert_text_line_with_indent("compare : process(clk)")
        text += self._insert_text_line_with_indent("begin")
        self.current_indent += 1
        text += self._insert_text_line_with_indent("if clk='1' and clk'event then")
        self.current_indent += 1
        text += self._insert_text_line_with_indent("if rst='1' then")
        text += self._insert_text_line_with_indent("elsif en='1' then")
        self.current_indent += 1

        # insert all splits
        for (split, i) in zip(self.splits, range(0, len(self.splits))):
            text += self._insert_text_line_with_indent(
                "if unsigned(features(" +
                str(split.var_idx) + ")) <= to_unsigned(" +
                str(int(split.value_to_compare)) +
                ", features'length) then")

            self.current_indent += 1
            text += self._insert_text_line_with_indent("splitResult(" + str(i) + ") <= '1';")
            self.current_indent -= 1
            text += self._insert_text_line_with_indent("else")
            self.current_indent += 1
            text += self._insert_text_line_with_indent("splitResult(" + str(i) + ") <= '0';")
            self.current_indent -= 1
            text += self._insert_text_line_with_indent("end if;")

        self.current_indent -= 1
        text += self._insert_text_line_with_indent("end if;")
        self.current_indent -= 1
        text += self._insert_text_line_with_indent("end if;")
        self.current_indent -= 1
        text += self._insert_text_line_with_indent("end process compare;")
        text += self._insert_text_line_with_indent("")

        return text

    def _add_architecture_process_decideClass(self):
        text = ""

         # create code for all the leaves
        text += self._insert_text_line_with_indent("decideClass : process(clk)")
        text += self._insert_text_line_with_indent("begin")
        self.current_indent += 1
        text += self._insert_text_line_with_indent("if clk='1' and clk'event then")
        self.current_indent += 1
        text += self._insert_text_line_with_indent("if rst='1' then")
        text += self._insert_text_line_with_indent("")
        text += self._insert_text_line_with_indent("elsif en='1' then")
        self.current_indent += 1

        for leaf in self.leaves:

            text += self._insert_text_line_with_indent("if ( ")
            self.current_indent += 1

            for split_id, split_compare_value in zip(leaf.following_split_IDs, leaf.following_split_compare_values):

                text += self._insert_text_line_with_indent(
                    "splitResult(" + str(split_id) + ") = '" + str(split_compare_value) + "'")

                if split_id != leaf.following_split_IDs[-1]:
                    text += self._insert_text_line_with_indent("and")
                else:
                    text += self._insert_text_line_with_indent(" ) then")
                    self.current_indent += 1

                    result = leaf.class_idx

                    # find the most important class
                    result_as_class = np.argmax(result[0])

                    text += self._insert_text_line_with_indent("classIndex <= to_unsigned(" + str(result_as_class) + ", classIndex'length);")
                    self.current_indent -= 1
                    text += self._insert_text_line_with_indent("end if;")
            self.current_indent -= 1

        self.current_indent -= 1
        text += self._insert_text_line_with_indent("end if;")
        self.current_indent -= 1
        text += self._insert_text_line_with_indent("end if;")
        self.current_indent -= 1
        text += self._insert_text_line_with_indent("end process decideClass;")

        return text

    def _add_new_leaf(self, id, class_idx, following_splits_IDs, following_splits_compare_values):
        new_leaf = Leaf(id, class_idx, following_splits_IDs, following_splits_compare_values)
        self.leaves.append(new_leaf)
        self._current_leaf_index += 1

    def _add_new_split(self, id, var_idx, value_to_compare):
        new_split = Split(id, var_idx, value_to_compare)
        self.splits.append(new_split)
        self._current_split_index += 1

    def _preorder(self, tree_, features, following_splits_IDs, following_splits_compare_values, node):

        # if the node is not the end of the path (it is not a leaf)
        if tree_.feature[node] != sklearn.tree._tree.TREE_UNDEFINED:
            following_splits_IDs.append(self._current_split_index)

            # use this to print the features before and after the conversion to fixed point
            #print("Feature: " + str(tree_.threshold[node]) + ", after conversion: " +
                  #str(convert_to_fixed_point(tree_.threshold[node], self._number_of_bits_per_feature)))

            # then create a split
            self._add_new_split(
                self._current_split_index,
                features[node],
                # the raw value has to be change to a fixed point with appropriate number of bits
                #tree_.threshold[node]
                convert_to_fixed_point(tree_.threshold[node], self._number_of_bits_per_feature)
            )

            # run the code for the left side of the tree (comparision wan not true,
            # hence 0 into list of following splits
            following_splits_compare_values.append(0)

            self._preorder(
                tree_,
                features,
                list(following_splits_IDs),
                list(following_splits_compare_values),
                tree_.children_left[node]
            )

            # now do something similar to the right side.
            # first remove appended 0 value (False)
            following_splits_compare_values.pop()
            # and add 1 (the comparision result is True)
            following_splits_compare_values.append(1)

            self._preorder(
                tree_,
                features,
                list(following_splits_IDs),
                list(following_splits_compare_values),
                tree_.children_right[node]
            )

        else:
            # otherwise the node is a leaf
            self._add_new_leaf(
                self._current_leaf_index,
                tree_.value[node],
                following_splits_IDs,
                following_splits_compare_values
            )
