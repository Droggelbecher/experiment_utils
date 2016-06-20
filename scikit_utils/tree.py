#!/usr/bin/env python

from collections import Counter
from sklearn.tree import _tree
from sklearn.externals import six
import itertools
import numpy as np


def leaf_iter(tree, feature_names=None, class_names=None, report = 'ft', include_leaf = False):
    """
    report:
        f -> feature
        t -> threshold
        i -> id
    """

    node_id = 0

    def recurse(node_id, path_):
        assert node_id < len(tree.children_left)
        assert node_id < len(tree.children_right)
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        if feature_names is not None:
            feature = feature_names[tree.feature[node_id]]
        else:
            feature = "X[%s]" % tree.feature[node_id]


        v = []
        for c in report:
            v.append({
                'i': node_id,
                't': tree.threshold[node_id],
                'f': feature
            }[c])

        path = path_ + (tuple(v), )

        if left_child == _tree.TREE_LEAF:
            if include_leaf:
                yield path
            else:
                yield path[:-1]
        else:
            for x in itertools.chain(recurse(left_child, path), recurse(right_child, path)):
                yield x


    return recurse(node_id, ())



def analyze_tree(decision_tree, feature_names=None, class_names=None):
    # for each path determine the feature with the maximum number of mentions

    max_path = []
    max_path_count = 0
    max_path_elem = ''

    for path in leaf_iter(decision_tree, feature_names, class_names):
        featcount = Counter()
        for feat, val in path:
            featcount[feat] += 1
        elem, count = featcount.most_common(1)[0]
        if count > max_path_count:
            max_path = path
            max_path_count = count
            max_path_elem = elem

    print("most suspect path: {}\nwith {}x'{}' ".format(str(max_path), max_path_count, max_path_elem))


def make_subtree(tree, leaf_ids):

    class SubTreeWrapper:
        def __init__(self, tree, leaf_ids):
            self.classes_ = tree.classes_
            self.criterion = tree.criterion

            tree = tree.tree_
            self.tree_ = self

            leaf_ids = set(leaf_ids)
            inner_ids = set()
            for path in leaf_iter(tree, report='i', include_leaf=True):
                if path[-1][0] in leaf_ids:
                    inner_ids.update(x[0] for x in path[:-1])

            ids = sorted(leaf_ids.union(inner_ids))

            index_transform = np.full(tree.node_count + 1, -2, dtype=np.int)
            for i, id_ in enumerate(ids):
                index_transform[id_] = i

            # trick to keep -1 (special value for marking leaves) unmapped
            index_transform[-1] = -1

            mask = np.zeros(len(index_transform) - 1, dtype=np.bool)
            mask[index_transform[:-1] != -2] = True

            self.children_left = index_transform[ tree.children_left[mask] ]

            self.children_right = index_transform[ tree.children_right[mask] ]
            self.feature = tree.feature[mask]
            self.threshold = tree.threshold[mask]
            self.value = tree.value[mask]
            self.impurity = tree.impurity[mask]
            self.n_outputs = tree.n_outputs
            self.n_node_samples = tree.n_node_samples[mask]

    return SubTreeWrapper(tree, leaf_ids)


def export_graphviz(decision_tree, out_file="tree.dot", feature_names=None,
                    max_depth=None, class_names=None, rankdir='LR', highlight_path=None):
    """Export a decision tree in DOT format.

    This function generates a GraphViz representation of the decision tree,
    which is then written into `out_file`. Once exported, graphical renderings
    can be generated using, for example::

        $ dot -Tps tree.dot -o tree.ps      (PostScript format)
        $ dot -Tpng tree.dot -o tree.png    (PNG format)

    The sample counts that are shown are weighted with any sample_weights that
    might be present.

    Parameters
    ----------
    decision_tree : decision tree classifier
        The decision tree to be exported to GraphViz.

    out_file : file object or string, optional (default="tree.dot")
        Handle or name of the output file.

    feature_names : list of strings, optional (default=None)
        Names of each of the features.

    max_depth : int, optional (default=None)
        The maximum depth of the representation. If None, the tree is fully
        generated.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn import tree

    >>> clf = tree.DecisionTreeClassifier()
    >>> iris = load_iris()

    >>> clf = clf.fit(iris.data, iris.target)
    >>> tree.export_graphviz(clf,
    ...     out_file='tree.dot')                # doctest: +SKIP
    """

    if class_names is None:
        class_names = { k: str(k) for k in decision_tree.classes_ }
        #class_names = [str(x) for x in range(len(decision_tree.classes_))]

    def pretty_samples(samples):
        s = ''
        if class_names is not None:
            for i, n in enumerate(samples):
                if n != 0:
                    s += '%s(%d) ' % (class_names[decision_tree.classes_[i]], int(n))

        else:
            s = str(samples)
        return s


    def node_to_str(tree, node_id, criterion):
        if not isinstance(criterion, six.string_types):
            criterion = "impurity"

        value = tree.value[node_id]
        if tree.n_outputs == 1:
            value = value[0, :]

        value = pretty_samples(value)

        if tree.children_left[node_id] == _tree.TREE_LEAF:
            return "(%d)\\n%s = %.4f\\nsamples = %s\\n%s" \
                   % (node_id, criterion,
                      tree.impurity[node_id],
                      tree.n_node_samples[node_id],
                      value)
        else:
            if feature_names is not None:
                feature = feature_names[tree.feature[node_id]]
            else:
                feature = "X[%s]" % tree.feature[node_id]

            return "(%d)\\n%s <= %.4f\\nsamples = %s" \
                   % (node_id,
                      feature,
                      tree.threshold[node_id],
                      tree.n_node_samples[node_id])

    highlighted_nodes = set([0])

    def recurse(tree, node_id, criterion, parent=None, depth=0, left=False):

        if node_id == _tree.TREE_LEAF:
            raise ValueError("Invalid node_id %s" % _tree.TREE_LEAF)

        if node_id == -2:
            return -2

        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        # Add node with description
        if max_depth is None or depth <= max_depth:
            is_leaf = (left_child == _tree.TREE_LEAF)

            fillcolor = '#ffffff'
            penwidth = 1

            if is_leaf:
                penwidth = 3

            i = tree.impurity[node_id]
            fillcolor = '#{:02x}{:02x}00'.format(
                    int(255 * min(1.0, 2.0 * i)),
                    int(255 * min(1.0, 2.0 - 2.0 * i))
                    )

            out_file.write('%d [label="%s", shape="box", style="filled", fillcolor="%s", penwidth=%d] ;\n' %
                           (node_id, node_to_str(tree, node_id, criterion), fillcolor, penwidth))

            if parent is not None:

                path_taken = False
                if highlight_path is not None and parent in highlighted_nodes:
                    v = highlight_path[tree.feature[parent]]
                    pivot = tree.threshold[parent]

                    path_taken = (left and (v <= pivot)) or (not left and v > pivot)
                    if path_taken:
                        highlighted_nodes.add(node_id)

                # Add edge to parent
                if left:
                    out_file.write('%d -> %d [label="<=",penwidth=%d];\n' % (parent, node_id, 5 if path_taken else 1))
                else:
                    out_file.write('%d -> %d [label=">",penwidth=%d];\n' % (parent, node_id, 5 if path_taken else 1))

            if not is_leaf:
                l = recurse(tree, left_child, criterion=criterion, parent=node_id,
                        depth=depth + 1, left=True)
                r = recurse(tree, right_child, criterion=criterion, parent=node_id,
                        depth=depth + 1)

                if l is None: return r
                return l

        else:
            out_file.write('%d [label="(...)", shape="box"] ;\n' % node_id)

            if parent is not None:
                # Add edge to parent
                out_file.write('%d -> %d ;\n' % (parent, node_id))

    own_file = False
    try:
        if isinstance(out_file, six.string_types):
            if six.PY3:
                out_file = open(out_file, "w", encoding="utf-8")
            else:
                out_file = open(out_file, "wb")
            own_file = True

        out_file.write("digraph Tree {\n")
        out_file.write("  rankdir=" + rankdir + ";\n")

        if isinstance(decision_tree, _tree.Tree):
            recurse(decision_tree, 0, criterion="impurity")
        else:
            recurse(decision_tree.tree_, 0, criterion=decision_tree.criterion)
        out_file.write("}")

    finally:
        if own_file:
            out_file.close()


if __name__ == '__main__':
    from sklearn.tree import DecisionTreeClassifier
    import numpy as np

    X = np.random.randint(0, 100, size=(100, 10))
    y = np.random.randint(0, 100, size=(100,))

    classifier = DecisionTreeClassifier(min_samples_leaf = 8)
    classifier.fit(X, y)

    export_graphviz(classifier, out_file='full.dot')

    subtree = make_subtree(classifier, np.random.randint(0, classifier.tree_.node_count, size=(3,)))

    export_graphviz(subtree, out_file='subtree.dot')


    #def export_graphviz(decision_tree, out_file="tree.dot", feature_names=None,
                        #max_depth=None, class_names=None, rankdir='LR', highlight_path=None):
    


