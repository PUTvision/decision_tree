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


class RandomForest:

    def __init__(self):
        self.random_forest = []

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

    def create_vhdl_code(self, filename):
        pass


class Tree:

    def __init__(self):
        self._current_split_index = 0
        self._current_leaf_index = 0

        self.splits = []
        self.leaves = []

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
    def _insert_text_line_with_indent(self, text_to_insert, current_indent):
        text = ""
        for i in xrange(0, current_indent):
            text += "\t"
        text += text_to_insert
        text += "\n"
        return text

    def create_vhdl_code(self, filename):
        f = open(filename, 'w')

        # create code for all compares done in splits
        text = ""
        current_indent = 0
        text += self._insert_text_line_with_indent("compare : process(clk)", current_indent)
        text += self._insert_text_line_with_indent("begin", current_indent)
        current_indent += 1
        text += self._insert_text_line_with_indent("if clk='1' and clk'event then", current_indent)
        current_indent += 1
        text += self._insert_text_line_with_indent("if rst='1' then", current_indent)
        text += self._insert_text_line_with_indent("elsif en='1' then", current_indent)
        current_indent += 1

        # insert all splits
        for (split, i) in zip(self.splits, xrange(0, len(self.splits))):
            text += self._insert_text_line_with_indent(
                "if unsigned(input(" +
                str(split.var_idx) + ")) > to_unsigned(" +
                str(int(split.value_to_compare)) +
                ", input'length) then",
                current_indent)

            current_indent += 1
            text += self._insert_text_line_with_indent("splitResult(" + str(i) + ") <= '1';", current_indent)
            current_indent -= 1
            text += self._insert_text_line_with_indent("else", current_indent)
            current_indent += 1
            text += self._insert_text_line_with_indent("splitResult(" + str(i) + ") <= '0';", current_indent)
            current_indent -= 1
            text += self._insert_text_line_with_indent("end if;", current_indent)

        current_indent -= 1
        text += self._insert_text_line_with_indent("end if;", current_indent)
        current_indent -= 1
        text += self._insert_text_line_with_indent("end if;", current_indent)
        current_indent -= 1
        text += self._insert_text_line_with_indent("end process compare;", current_indent)
        text += self._insert_text_line_with_indent("", current_indent)

        f.write(text)

        text = ""
        current_indent = 0

        # create code for all the leaves
        text += self._insert_text_line_with_indent("decideClass : process(clk)", current_indent)
        text += self._insert_text_line_with_indent("begin", current_indent)
        current_indent += 1
        text += self._insert_text_line_with_indent("if clk='1' and clk'event then", current_indent)
        current_indent += 1
        text += self._insert_text_line_with_indent("if rst='1' then", current_indent)
        text += self._insert_text_line_with_indent("", current_indent)
        text += self._insert_text_line_with_indent("elsif en='1' then", current_indent)
        current_indent += 1

        for leaf in self.leaves:

            text += self._insert_text_line_with_indent("if ( ", current_indent)
            current_indent += 1

            for split_id, split_compare_value in zip(leaf.following_split_IDs, leaf.following_split_compare_values):

                text += self._insert_text_line_with_indent(
                    "splitResult(" + str(split_id) + ") = '" + str(split_compare_value) + "'", current_indent)

                if split_id != leaf.following_split_IDs[-1]:
                    #if(j < currentLeaf->listOfFollowingSplitsIDs.size()-1)
                    text += self._insert_text_line_with_indent("and", current_indent)
                else:
                    text += self._insert_text_line_with_indent(" ) then", current_indent)
                    current_indent += 1

                    result = leaf.class_idx

                    if result[0][0] > result[0][1]:
                        result_as_class = 0
                    else:
                        result_as_class = 1

                    text += self._insert_text_line_with_indent("classIndex <= to_unsigned(" + str(result_as_class) + ", classIndex'length);", current_indent)
                    #// earlier version
                    #//myfile << "\t\t\t\toutput(" << i << ") <= '1';" << endl;
                    #//myfile << "\t\t\telse" << endl;
                    #//myfile << "\t\t\t\toutput(" << i << ") <= '0';" << endl;
                    current_indent -= 1
                    text += self._insert_text_line_with_indent("end if;", current_indent)
            current_indent -= 1


        current_indent -= 1
        text += self._insert_text_line_with_indent("end if;", current_indent)
        current_indent -= 1
        text += self._insert_text_line_with_indent("end if;", current_indent)
        current_indent -= 1
        text += self._insert_text_line_with_indent("end process decideClass;", current_indent)
        f.write(text)

        f.close()

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
