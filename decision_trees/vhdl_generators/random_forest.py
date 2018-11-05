from typing import List

from decision_trees.vhdl_generators.VHDLCreator import VHDLCreator
from decision_trees.vhdl_generators.tree import Tree

import numpy as np
import sklearn.ensemble

from decision_trees.utils.constants import ClassifierType


class RandomForest(VHDLCreator):

    def __init__(self, name: str, number_of_features: int, number_of_bits_per_feature: int):
        self.random_forest: List[Tree] = []

        VHDLCreator.__init__(self, name, ClassifierType.RANDOM_FOREST.name,
                             number_of_features, number_of_bits_per_feature)

    def build(self, random_forest: sklearn.ensemble.RandomForestClassifier):
        for i, tree in enumerate(random_forest.estimators_):
            tree_builder = Tree(f'tree_{i:02}', self._number_of_features, self._number_of_bits_per_feature)
            tree_builder.build(tree)

            self.random_forest.append(tree_builder)

    # TODO(MF): this could probably be moved as a common element for random forest and decision tree
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        result_data = np.empty(len(input_data))

        for i in range(len(input_data)):
            result_data[i] = self._predict_one_sample(input_data[i])

        return result_data

    def _predict_one_sample(self, input_data: np.ndarray) -> int:
        # first create a dictionary that will store the results
        results = {}
        for tree in self.random_forest:
            # get the result form one of the tree and add it to appropriate element in dict
            tree_result = tree._predict_one_sample(input_data)
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
        print(f"Number of decision trees: {len(self.random_forest)}")
        sum_depth = 0
        number_of_splits = 0
        number_of_leaves = 0
        number_of_decide_class_compares = 0
        for tree in self.random_forest:
            sum_depth += tree.find_depth()
            number_of_splits += len(tree.splits)
            number_of_leaves += len(tree.leaves)
            number_of_decide_class_compares += tree.decide_class_compares
            # tree.print_parameters()
        print("avg depth: ", sum_depth / len(self.random_forest))
        print("avg number of splits: ", number_of_splits / len(self.random_forest))
        print("avg number of leaves: ", number_of_leaves / len(self.random_forest))
        print("avg decide_class_compares: ", number_of_decide_class_compares / len(self.random_forest))

    def _add_additional_headers(self) -> str:
        text = ""
        return text

    def _add_entity_generics_section(self) -> str:
        text = ""
        return text

    def _add_architecture_component_section(self) -> str:
        text = ""

        for i in range(0, len(self.random_forest)):
            text += self._insert_text_line_with_indent(f"component tree_{i:02}")
            text += self._add_entity_generics_section()
            text += self._add_entity_port_section()
            text += self._insert_text_line_with_indent(f"end component tree_{i:02};")
            text += self._insert_text_line_with_indent("")

        # TODO(MF): add module for connecting the results

        return text

    def _add_architecture_signal_section(self) -> str:
        text = ""

        text += self._insert_text_line_with_indent(f"type outputs_t\tis array({len(self.random_forest)}-1 downto 0)" +
                                                   f" of std_logic_vector({self._number_of_bits_for_class_index}-1 downto 0);")

        text += self._insert_text_line_with_indent("signal " + "outputs" + "\t\t:\t" + "outputs_t"
                                                   + "\t\t\t" + ":= (others=>(others=>'0'));")

        return text

    def _add_architecture_process_section(self) -> str:
        text = ''

        for i in range(0, len(self.random_forest)):
            text += self._add_port_mapping(i)

        # TODO(MF): add Marek's module for combinig the results
        for i in range(0, len(self.random_forest)):
            text += self._insert_text_line_with_indent(
                f'output({i*self._number_of_bits_for_class_index+self._number_of_bits_for_class_index-1} downto'
                f' {i*self._number_of_bits_for_class_index}) <= std_logic_vector(outputs({i}));'
            )
        text += self._insert_text_line_with_indent("")

        return text

    def _add_port_mapping(self, index: int) -> str:
        text = ""

        text += self._insert_text_line_with_indent(
            f"{ClassifierType.DECISION_TREE.name}_{index}_INST : tree_{index:02}"
        )
        text += self._insert_text_line_with_indent("port map (")
        self.current_indent += 2

        text += self._insert_text_line_with_indent("clk => clk,")
        text += self._insert_text_line_with_indent("rst => rst,")
        text += self._insert_text_line_with_indent("en => en,")
        text += self._insert_text_line_with_indent("input => input,")
        text += self._insert_text_line_with_indent(f"output => outputs({index})")

        self.current_indent -= 1
        text += self._insert_text_line_with_indent(");")
        self.current_indent -= 1

        return text

    def create_vhdl_file(self, path: str):
        for d in self.random_forest:
            d.create_vhdl_file(path)

        with open(path + '/' + self._filename, 'w') as f:
            text = ''
            text += self._add_headers()
            text += self._add_entity(len(self.random_forest))
            text += self._add_architecture()
            f.write(text)

