from decision_trees.vhdl_generators.VHDLCreator import VHDLCreator

import numpy as np
import sklearn.tree

from decision_trees.utils.convert_to_fixed_point import convert_to_fixed_point, convert_fixed_point_to_integer
from decision_trees.utils.constants import ClassifierType


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

    def __init__(self, id_, class_idx_, following_split_IDs_, following_split_compare_values_):
        self.id = id_
        self.class_idx = class_idx_
        self.following_split_IDs = following_split_IDs_
        self.following_split_compare_values = following_split_compare_values_

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

class Tree(VHDLCreator):

    def __init__(self, name: str, number_of_features: int, number_of_bits_per_feature: int):
        self._current_split_index = 0
        self._current_leaf_index = 0

        self.splits = []
        self.leaves = []
        self.decide_class_compares = 0

        VHDLCreator.__init__(self, name, name,
                             number_of_features, number_of_bits_per_feature)

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
        print("decide_class_compares: ", self.decide_class_compares)

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

        text += self._insert_text_line_with_indent("signal " + "sr" + "\t:\t" + "std_logic_vector("
                                                   + str(len(self.splits)) + "-1 downto 0)"
                                                   + "\t\t\t" + ":= (others=>'0');")
        text += self._insert_text_line_with_indent("signal " + "ci" + "\t:\t" + "unsigned("
                                                   + str(self._number_of_bits_for_class_index) + "-1 downto 0)"
                                                   + "\t\t\t" + ":= (others=>'0');")

        text += self._insert_text_line_with_indent("")

        return text

    def _add_architecture_process_section(self):
        text = ""

        text += self._add_architecture_input_mapping()
        text += self._add_architecture_process_compare()
        text += self._add_architecture_process_decide_class()
        text += self._insert_text_line_with_indent("output <= std_logic_vector(ci);")
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
                str(convert_fixed_point_to_integer(split.value_to_compare, self._number_of_bits_per_feature)) +
                ", features'length) then")

            self.current_indent += 1
            text += self._insert_text_line_with_indent("sr(" + str(i) + ") <= '0';")
            self.current_indent -= 1
            text += self._insert_text_line_with_indent("else")
            self.current_indent += 1
            text += self._insert_text_line_with_indent("sr(" + str(i) + ") <= '1';")
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

    def _add_architecture_process_decide_class(self):
        text = ""

        # create the code for all the leaves
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

            # text += self._insert_text_line_with_indent("if ( ")
            self.current_indent += 1
            compare_line = "if ( "

            for split_id, split_compare_value in zip(leaf.following_split_IDs, leaf.following_split_compare_values):

                self.decide_class_compares += 1

                # text += self._insert_text_line_with_indent(
                compare_line += "sr(" + str(split_id) + ") = '" + str(split_compare_value) + "'" #)

                if split_id != leaf.following_split_IDs[-1]:
                    # text += self._insert_text_line_with_indent("and")
                    compare_line += " and "
                else:
                    # text += self._insert_text_line_with_indent(" ) then")
                    compare_line += " ) then"
                    text += self._insert_text_line_with_indent(compare_line)
                    self.current_indent += 1

                    result = leaf.class_idx

                    # find the most important class
                    result_as_class = np.argmax(result[0])

                    text += self._insert_text_line_with_indent(
                        "ci <= to_unsigned(" + str(result_as_class) + ", ci'length);"
                    )
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
            # print("Feature: " + str(tree_.threshold[node]) + ", after conversion: "
            # + str(convert_to_fixed_point(tree_.threshold[node], self._number_of_bits_per_feature)))

            #print(f'tree_.threshold[node]: {tree_.threshold[node]:{1}.{3}}')
            #print(f'convert_to_fixed_point: {convert_to_fixed_point(tree_.threshold[node], self._number_of_bits_per_feature):{1}.{3}}')

            # then create a split
            self._add_new_split(
                self._current_split_index,
                features[node],
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

    def create_vhdl_file(self, path: str):
        with open(path + '/' + self._filename, 'w') as f:
            text = ''
            text += self._add_headers()
            text += self._add_entity()
            text += self._add_architecture()
            f.write(text)
