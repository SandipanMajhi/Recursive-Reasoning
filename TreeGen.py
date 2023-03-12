from graphviz import Digraph

class TreeGen:
  def __init__(self, root, node_dict, node_decisions, sat_solution, belief_scores, consistency_scores):
    self.root_node = root
    self.node_decisions = node_decisions
    self.solution = sat_solution
    self.node_dict = None
    self.node_list = node_dict
    self.belief_scores = belief_scores
    self.consistency_scores = consistency_scores
    self.build_node_dict()

  def build_node_dict(self):
    if self.node_dict == None:
      self.node_dict = {self.node_list[index] : index+1 for index in range(len(self.node_list))}


  def nodenamer(self,node):
    if node.name in self.belief_scores:
      return "%s Score = %.3f" % (node.name, self.belief_scores[node.name].item())
    else:
      return "%s" % (node.name)
  
  def nodeformatter(self,node):
    attrs = []
    attrs += [f'shape=box']
    attrs += [f'fontsize=10']
    if self.node_dict[node.name] in self.solution:
      attrs += [f'color=blue']
    elif -1 * self.node_dict[node.name] in self.solution:
      attrs += [f'color=brown']
    return ", ".join(attrs)

  def edgeformatter(self,parent,child):
    attrs = []
    if child.name in self.node_decisions:
      if self.node_decisions[child.name] == "True":
        attrs += [f'color=green']
      else:
        attrs += [f'color=red']
    attrs += [f'label = {"%.3f" % self.consistency_scores[(parent.name, child.name, self.node_decisions[child.name])]}']
    return ", ".join(attrs)

  def runner(self):
     DotExporter(node = self.root_node, graph = "digraph", nodeattrfunc= self.nodeformatter, options = ['rankdir = "LR"'],
                nodenamefunc=self.nodenamer, edgeattrfunc = self.edgeformatter).to_dotfile("tree_sample.dot")