__author__ = 'Amin'


class Split:

    id = 0
    var_idx = 0
    value_to_compare = 0.0

    def __init__(self, id_, var_idx_, value_to_compare_):
        self.id = id_
        self.var_idx = var_idx_
        self.value_to_compare = value_to_compare_

    def show(self):
        print "Split[", self.id, "], var_idx: ", self.var_idx, ", value to compare: ", self.value_to_compare


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
        print "Leaf[", self.id, "], class_idx: ", self.class_idx

        print "Following splits IDs: "
        splits_ids = ""
        for id_ in self.following_split_IDs:
            splits_ids += str(id_) + ", "
        print splits_ids

        print "Following split compare values: "
        compare_values = ""
        for compare_value in self.following_split_compare_values:
            compare_values += str(compare_value) + ", "
        print compare_values

from abc import ABCMeta, abstractmethod


class VHDLcreator():

    def __init__(self, name):
        __metaclass__ = ABCMeta

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

    def _insert_text_line_with_indent(self, text_to_insert):
        text = ""
        for i in range(0, self.current_indent):
            text += "\t"
        text += text_to_insert
        text += "\n"
        return text

    def _add_headers(self):
        text = ""
        text += self._insert_text_line_with_indent("library IEEE;")
        text += self._insert_text_line_with_indent("use IEEE.STD_LOGIC_1164.ALL;")
        text += self._insert_text_line_with_indent("use IEEE.NUMERIC_STD.ALL;")
        text += self._insert_text_line_with_indent("")

        text += self.add_additional_headers()
        text += self._insert_text_line_with_indent("")

        return text

    @abstractmethod
    def add_additional_headers(self):
        pass

    def _add_entity(self):
        text = ""

        text += self._insert_text_line_with_indent("entity " + self._param_entity_name + " is")
        text += self._add_entity_generics_section()
        text += self._add_entity_port_section()
        text += self._insert_text_line_with_indent("end " + self._param_entity_name + ";")
        text += self._insert_text_line_with_indent("")

        return text

    @abstractmethod
    def _add_entity_generics_section(self):
        pass

    @abstractmethod
    def _add_entity_port_section(self):
        pass

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

    @abstractmethod
    def _add_architecture_component_section(self):
        pass


    @abstractmethod
    def _add_architecture_signal_section(self):
        pass

    @abstractmethod
    def _add_architecture_process_section(self):
        pass

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

    def __init__(self):
        self.random_forest = []

        VHDLcreator.__init__(self, "RandomForestTest")

    def build(self, random_forest, feature_names):
        for tree in random_forest.estimators_:
            tree_builder = Tree()
            tree_builder.build(tree, feature_names)

            self.random_forest.append(tree_builder)

    def predict(self, input_data):
        # TODO - make this fragment universal (to work for more than two classes)
        results = [0, 0]
        for tree in self.random_forest:
            tree_result = tree.predict(input_data)
            results[int(tree_result)] += 1

        if results[0] >= results[1]:
            chosen_class = 0
        else:
            chosen_class = 1

        return chosen_class

    def print_parameters(self):
        for tree in self.random_forest:
            tree.print_parameters()

    def create_vhdl_code_old(self, filename):
        self.create_vhdl_file()

    def add_additional_headers(self):
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

    def __init__(self):
        self._current_split_index = 0
        self._current_leaf_index = 0

        self.splits = []
        self.leaves = []

        VHDLcreator.__init__(self, "TreeTest")

    def build(self, tree, feature_names):
        self._current_split_index = 0
        self._current_leaf_index = 0

        self.splits = []
        self.leaves = []

        following_splits_IDs = []
        following_splits_compare_values = []

        features = [feature_names[i] for i in tree.tree_.feature]

        left = tree.tree_.children_left
        right = tree.tree_.children_right
        threshold = tree.tree_.threshold
        value = tree.tree_.value

        self._preorder(left, right, threshold, value, features, following_splits_IDs, following_splits_compare_values, 0)

    def predict(self, input_data):
        choosen_class_index = [-1, -1]

        compare_results = [None] * len(self.splits)

        for split, i in zip(self.splits, xrange(0, len(self.splits))):
            if input_data[split.var_idx] <= split.value_to_compare:
                compare_results[i] = 0
            else:
                compare_results[i] = 1

        #print compare_results

        for leaf in self.leaves:
            number_of_correct_results = 0

            for j in xrange(0, len(leaf.following_split_IDs)):
                split_id = leaf.following_split_IDs[j]
                expected_result = leaf.following_split_compare_values[j]
                real_result = compare_results[split_id]

                #print "Expected result: ", expected_result, ", real result: ", real_result

                if expected_result == real_result:
                    number_of_correct_results += 1

            #print "Number of correct result: ", number_of_correct_results, ", should be: ", len(leaf.following_split_IDs)

            if number_of_correct_results == len(leaf.following_split_IDs):
                choosen_class_index = leaf.class_idx

        # TODO - make this fragment universal (to work for more than two classes)
        if choosen_class_index[0][0] > choosen_class_index[0][1]:
            chosen_class = 0
        else:
            chosen_class = 1

        return chosen_class

    def print_parameters(self):
        #self.print_leaves()
        #self.print_splits()
        print "Depth: ", self.find_depth()
        print "Number of splits: ", len(self.splits)
        print "Number of leaves: ", len(self.leaves)

    def print_splits(self):
        print "Splits: "
        print len(self.splits)
        for split in self.splits:
            split.show()

    def print_leaves(self):
        print "Leaves: "
        print len(self.leaves)
        for leaf in self.leaves:
            leaf.show()

    def find_depth(self):

        following_splits_number = []

        for leaf in self.leaves:
            following_splits_number.append(len(leaf.following_split_IDs))

        return max(following_splits_number)

    @staticmethod
    def _insert_text_line_with_indent_old(self, text_to_insert, current_indent):
        text = ""
        for i in xrange(0, current_indent):
            text += "\t"
        text += text_to_insert
        text += "\n"
        return text

    def create_vhdl_code_old(self, filename):
        f = open(filename, 'w')

        # create code for all compares done in splits
        text = ""
        current_indent = 0
        text += self._insert_text_line_with_indent_old("compare : process(clk)", current_indent)
        text += self._insert_text_line_with_indent_old("begin", current_indent)
        current_indent += 1
        text += self._insert_text_line_with_indent_old("if clk='1' and clk'event then", current_indent)
        current_indent += 1
        text += self._insert_text_line_with_indent_old("if rst='1' then", current_indent)
        text += self._insert_text_line_with_indent_old("elsif en='1' then", current_indent)
        current_indent += 1

        # insert all splits
        for (split, i) in zip(self.splits, xrange(0, len(self.splits))):
            text += self._insert_text_line_with_indent_old(
                "if unsigned(input(" +
                str(split.var_idx) + ")) > to_unsigned(" +
                str(int(split.value_to_compare)) +
                ", input'length) then",
                current_indent)

            current_indent += 1
            text += self._insert_text_line_with_indent_old("splitResult(" + str(i) + ") <= '1';", current_indent)
            current_indent -= 1
            text += self._insert_text_line_with_indent_old("else", current_indent)
            current_indent += 1
            text += self._insert_text_line_with_indent_old("splitResult(" + str(i) + ") <= '0';", current_indent)
            current_indent -= 1
            text += self._insert_text_line_with_indent_old("end if;", current_indent)

        current_indent -= 1
        text += self._insert_text_line_with_indent_old("end if;", current_indent)
        current_indent -= 1
        text += self._insert_text_line_with_indent_old("end if;", current_indent)
        current_indent -= 1
        text += self._insert_text_line_with_indent_old("end process compare;", current_indent)
        text += self._insert_text_line_with_indent_old("", current_indent)

        f.write(text)

        text = ""
        current_indent = 0

        # create code for all the leaves
        text += self._insert_text_line_with_indent_old("decideClass : process(clk)", current_indent)
        text += self._insert_text_line_with_indent_old("begin", current_indent)
        current_indent += 1
        text += self._insert_text_line_with_indent_old("if clk='1' and clk'event then", current_indent)
        current_indent += 1
        text += self._insert_text_line_with_indent_old("if rst='1' then", current_indent)
        text += self._insert_text_line_with_indent_old("", current_indent)
        text += self._insert_text_line_with_indent_old("elsif en='1' then", current_indent)
        current_indent += 1

        for leaf in self.leaves:

            text += self._insert_text_line_with_indent_old("if ( ", current_indent)
            current_indent += 1

            for split_id, split_compare_value in zip(leaf.following_split_IDs, leaf.following_split_compare_values):

                text += self._insert_text_line_with_indent_old(
                    "splitResult(" + str(split_id) + ") = '" + str(split_compare_value) + "'", current_indent)

                if split_id != leaf.following_split_IDs[-1]:
                    #if(j < currentLeaf->listOfFollowingSplitsIDs.size()-1)
                    text += self._insert_text_line_with_indent_old("and", current_indent)
                else:
                    text += self._insert_text_line_with_indent_old(" ) then", current_indent)
                    current_indent += 1

                    result = leaf.class_idx

                    if result[0][0] > result[0][1]:
                        result_as_class = 0
                    else:
                        result_as_class = 1

                    text += self._insert_text_line_with_indent_old("classIndex <= to_unsigned(" + str(result_as_class) + ", classIndex'length);", current_indent)
                    #// earlier version
                    #//myfile << "\t\t\t\toutput(" << i << ") <= '1';" << endl;
                    #//myfile << "\t\t\telse" << endl;
                    #//myfile << "\t\t\t\toutput(" << i << ") <= '0';" << endl;
                    current_indent -= 1
                    text += self._insert_text_line_with_indent_old("end if;", current_indent)
            current_indent -= 1


        current_indent -= 1
        text += self._insert_text_line_with_indent_old("end if;", current_indent)
        current_indent -= 1
        text += self._insert_text_line_with_indent_old("end if;", current_indent)
        current_indent -= 1
        text += self._insert_text_line_with_indent_old("end process decideClass;", current_indent)
        f.write(text)

        f.close()

    def add_additional_headers(self):
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
        # TODO - calc the input vector size automatically
        text += self._insert_text_line_with_indent("input" + "\t\t\t" + ":" + "\t" + "in std_logic_vector("
                                                   + str(6000*8) + "-1 downto 0);")
        # output is 0-1 at the moment
        # TODO - change to more general solution with more classes
        text += self._insert_text_line_with_indent("output" + "\t\t\t" + ":" + "\t" + "out std_logic")

        self.current_indent -= 1

        text += self._insert_text_line_with_indent(");")

        self.current_indent -= 1

        return text

    def _add_architecture_component_section(self):
        text = ""
        return text

    def _add_architecture_signal_section(self):
        text = ""

        # TODO - maybe add a function for creating signals?
        text += self._insert_text_line_with_indent("signal " + "splitResult" + "\t:\t" + "std_logic_vector("
                                                   + str(len(self.splits)) + "-1 downto 0)"
                                                   + "\t\t\t" + ":= (others=>'0');")
        text += self._insert_text_line_with_indent("signal " + "classIndex" + "\t:\t" + "std_logic_vector("
                                                   + str(1) + "-1 downto 0)"
                                                   + "\t\t\t" + ":= (others=>'0');")

        text += self._insert_text_line_with_indent("")

        return text

    def _add_architecture_process_section(self):
        text = ""

        text += self._add_architecture_process_compare()
        text += self._add_architecture_process_decideClass()
        text += self._insert_text_line_with_indent("output <= std_logic_vector(classIndex)")
        text += self._insert_text_line_with_indent("")

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
        for (split, i) in zip(self.splits, xrange(0, len(self.splits))):
            text += self._insert_text_line_with_indent(
                "if unsigned(input(" +
                str(split.var_idx) + ")) > to_unsigned(" +
                str(int(split.value_to_compare)) +
                ", input'length) then")

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

                    if result[0][0] > result[0][1]:
                        result_as_class = 0
                    else:
                        result_as_class = 1

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

    def _preorder(self, left, right, threshold, value, features, following_splits_IDs, following_splits_compare_values, node):

        if threshold[node] != -2:
            following_splits_IDs.append(self._current_split_index)

            self._add_new_split(
                self._current_split_index,
                features[node],
                threshold[node]
            )

            following_splits_compare_values.append(0)

            self._preorder(
                left,
                right,
                threshold,
                value,
                features,
                list(following_splits_IDs),
                list(following_splits_compare_values),
                left[node]
            )

            following_splits_compare_values.pop()
            following_splits_compare_values.append(1)

            self._preorder(
                left,
                right,
                threshold,
                value,
                features,
                list(following_splits_IDs),
                list(following_splits_compare_values),
                right[node]
            )

        else:
            self._add_new_leaf(
                self._current_leaf_index,
                value[node],
                following_splits_IDs,
                following_splits_compare_values
            )
