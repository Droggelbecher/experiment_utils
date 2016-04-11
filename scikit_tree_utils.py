#!/usr/bin/env python

from sklearn.tree import _tree
from sklearn.externals import six

def leaf_iter(tree, feature_names=None, class_names=None):

    node_id = 0

    def recurse(node_id, path=[]):
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        if feature_names is not None:
            feature = feature_names[tree.feature[node_id]]
        else:
            feature = "X[%s]" % tree.feature[node_id]

        path = path + [(feature, tree.threshold[node_id])]

        if left_child == _tree.TREE_LEAF:
            yield path
        else:
            recurse(left_child, path)
            recurse(right_child, path)



def analyze_tree(decision_tree, feature_names=None, class_names=None):

    # for each path determine the feature with the maximum number of mentions

    max_path = []
    max_path_count = 0

    for path in leaf_iter(decision_tree, feature_names, class_names):

        featcount = {}
        for feat, val in path:
            featcount[feat] = featcount.get(feat, 0) + 1
        count = max(featcount.values)
        if count > max_path_count:
            max_path = path
            max_path_count = count

    print(max_path_count)
    print(max_path)



def export_graphviz(decision_tree, out_file="tree.dot", feature_names=None,
                    max_depth=None, class_names=None, rankdir='LR'):
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

    def pretty_samples(samples):
        s = ''
        if class_names is not None:
            print(type(samples), samples)
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
            return "%s = %.4f\\nsamples = %s\\nvalue = %s" \
                   % (criterion,
                      tree.impurity[node_id],
                      tree.n_node_samples[node_id],
                      value)
        else:
            if feature_names is not None:
                feature = feature_names[tree.feature[node_id]]
            else:
                feature = "X[%s]" % tree.feature[node_id]

            return "%s <= %.4f\\n%s = %s\\nsamples = %s" \
                   % (feature,
                      tree.threshold[node_id],
                      criterion,
                      tree.impurity[node_id],
                      tree.n_node_samples[node_id])

    def recurse(tree, node_id, criterion, parent=None, depth=0):
        if node_id == _tree.TREE_LEAF:
            raise ValueError("Invalid node_id %s" % _tree.TREE_LEAF)

        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        # Add node with description
        if max_depth is None or depth <= max_depth:
            out_file.write('%d [label="%s", shape="box"] ;\n' %
                           (node_id, node_to_str(tree, node_id, criterion)))

            if parent is not None:
                # Add edge to parent
                out_file.write('%d -> %d ;\n' % (parent, node_id))

            if left_child != _tree.TREE_LEAF:
                recurse(tree, left_child, criterion=criterion, parent=node_id,
                        depth=depth + 1)
                recurse(tree, right_child, criterion=criterion, parent=node_id,
                        depth=depth + 1)

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

