from androguard.misc import AnalyzeAPK
from androguard.core.analysis.analysis import ExternalMethod
import matplotlib.pyplot as plt
import networkx as nx

a, d, dx = AnalyzeAPK("c:\\tmp\\b.apk")

CFG = nx.DiGraph()

# Note: If you create the CFG from many classes at the same time, the drawing
# will be a total mess...

for n in dx.find_classes('.*StartWebViewClient;'):
    n.get_vm_class()

for m in dx.find_methods(classname="Ldirxion/mobile/dyp/StartWebView;"):
    orig_method = m.get_method()
    print("Found Method --> {}".format(orig_method))
    # orig_method might be a ExternalMethod too...
    # so you can check it here also:
    if isinstance(orig_method, ExternalMethod):
        is_this_external = True
        # If this class is external, there will be very likely
        # no xref_to stored! If there is, it is probably a bug in androguard...
    else:
        is_this_external = False

    CFG.add_node(orig_method, external=is_this_external)

    for other_class, callee, offset in m.get_xref_to():
        print("calle Method --> {}".format(callee))
        if isinstance(callee, ExternalMethod):
            is_external = True
        else:
            is_external = False

        if callee not in CFG.node:
            CFG.add_node(callee, external=is_external)

        # As this is a DiGraph and we are not interested in duplicate edges,
        # check if the edge is already in the edge set.
        # If you need all calls, you probably want to check out MultiDiGraph
        if not CFG.has_edge(orig_method, callee):
            CFG.add_edge(orig_method, callee)

nx.draw_networkx(CFG)
plt.draw()
plt.show()