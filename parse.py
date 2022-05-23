from parsimonious.grammar import Grammar
from parsimonious.nodes import NodeVisitor
import psycopg2
import plotly.graph_objects as go
import os
import pickle
from tqdm import tqdm


DB_HOST = ''
DB_USER = ''
DB_PASS = ''
DB_PORT = 5432
DB_DATABASE = ''

# grammar = Grammar(
#     r"""
#     Start      = ws Node ws
#     Node       = lnode NodeName ws NodeItems? ws rnode
#     NodeName   = ~r"\w*"
#     NodeItems  = NodeItem (ws NodeItem)*
#     NodeItem   = Node / NodeText
#     NodeText   = ~r":\w*"

#     ws         = ~r"\s*"
#     lnode      = "{"
#     rnode      = "}"
#     lpar       = "\("
#     rpar       = "\)"
#     """
# )
def get_grammar():
    return Grammar(
        r"""
        Start        = ws Node ws

        Node         = lnode NodeName ws NodeItems? ws rnode
        NodeName     = ~r"\w*"
        NodeItems    = NodeItem (ws NodeItem)*
        NodeItem     = NodeKey (ws NodeValue ws)*
        NodeKey      = ~r":\w*"
        NodeValue    = Node / NodeArray / NodeList / Text / Number
        NodeArray    = ~r"\[\s*" ArrayItems? ~r"\s*\]"
        NodeList     = ~r"\(\s*" ArrayItems? ~r"\s*\)"

        ArrayItems   = ArrayValue (ws ArrayValue)*
        ArrayValue   = Node / NodeArray / NodeList / Text / Number

        Text         = ~r"(\w|\.|-|<|>|\\|\"|\$|\?|@)*"
        Number       = ~r"-?(0|[1-9][0-9]*)(\.\d*)?([eE][-+]?\d+)?"

        ws           = ~r"\s*"
        lnode        = "{"
        rnode        = "}"
        lpar         = "\("
        rpar         = "\)"
        """
    )

class Node():
    def __init__(self, attrs=None, edges=None, cost=0):
        if attrs is None:
            attrs = []
        if edges is None:
            edges = set()
        self.attrs = attrs
        self.edges = edges
        self.cost = cost

    def out(self):
        return {
            'edges': self.edges,
            'attrs': self.attrs,
            'cost': self.cost
        }

class pgVisitor(NodeVisitor):
    def __init__(self):
        self.graph = {}
        self.i = 0
        super().__init__()

    def generic_visit(self, node, children):
        return children or node.text

    def visit_Start(self, _, children):
        return children

    def visit_NodeName(self, node, _):
        self.i += 1
        return node.text + '_' + str(self.i)

    def visit_Node(self, _, children):
        _, node_name, _, *node_items, _, _ = children
        node_items = node_items[0][0]

        node_info = {
            'node_name': node_name,
            'attrs': node_items['attrs'],
            'edges': node_items['edges'],
            'cost': max(
                float(node_items['attrs'].get(':total_cost', [0.])[0]),
                node_items['cost']
            )
        }
        self.graph[node_name] = {k: v for k, v in node_info.items()}
        node_info['edges'] = set()
        return node_info

    def visit_NodeItems(self, _, children):
        node_items = [children[0], *children[1]]

        n = Node()
        for node_item in node_items:
            node_item = node_item[1] if isinstance(node_item, list) else node_item
            n.attrs.append(node_item['attrs'])
            n.edges.update(node_item['edges'])
            n.cost += node_item['cost']

        n.attrs = {key: value for key, value in [next(iter(v.items())) for v in n.attrs]}

        return n.out()

    def visit_NodeItem(self, _, children):
        node_key, node_values = children

        n = Node()
        for _, item, _ in node_values:
            n.attrs.append(item['attrs'])
            n.edges.update(item['edges'])
            n.cost += item['cost']

        n.attrs = {node_key: n.attrs}

        return n.out()

    def visit_NodeValue(self, _, children):
        item = children[0]

        n = Node(attrs=item['attrs'])
        if 'node_name' in item:
            n.edges.add(item['node_name'])
            n.attrs = item['node_name']
        else:
            n.edges.update(item['edges'])

        n.cost += item['cost']
        return n.out()

    def visit_NodeKey(self, node, _):
        return node.text

    def visit_NodeArray(self, _, children):
        _, (arr, *_), _ = children

        n = Node(edges=arr['edges'], attrs=arr['attrs'], cost=arr['cost'])
        return n.out()

    def visit_NodeList(self, _, children):
        _, (arr, *_), _ = children

        n = Node(edges=arr['edges'], attrs=arr['attrs'], cost=arr['cost'])
        return n.out()

    def visit_ArrayItems(self, _, children):
        arr_values = [children[0], *children[1]]

        n = Node()
        for item in arr_values:
            y = item[1] if isinstance(item, list) else item
            n.attrs.append(y['attrs'])
            n.edges.update(y['edges'])
            n.cost += y['cost']
        return n.out()

    def visit_ArrayValue(self, _, children):
        item = children[0]

        n = Node(edges=item['edges'], attrs=item['attrs'])
        if 'node_name' in item:
            n.attrs = [item['node_name']]
            n.edges.add(item['node_name'])
            n.cost += item['cost']

        return n.out()

    def visit_Text(self, node, _):
        n = Node(attrs=node.text)
        return n.out()

    def visit_Number(self, node, _):
        n = Node(attrs=float(node.text))
        return n.out()

    def visit_ws(self, *_):
        return None

    def visit_lnode(self, *_):
        return None

    def visit_rnode(self, *_):
        return None

def build_graph(log_path=None, log_string=None):
    if log_path is not None:
        with open(log_path, 'r') as f:
            txt = f.read()
    elif log_string is not None:
        txt = log_string

    tree = get_grammar().parse(txt)
    iv = pgVisitor()
    iv.visit(tree)
    graph = iv.graph

    return graph




def get_db_connection(database=DB_DATABASE):
    host = DB_HOST
    user = DB_USER
    password = DB_PASS
    port = DB_PORT

    # Load data:
    conn = psycopg2.connect(
        host=host,
        database=database,
        user=user,
        password=password,
        port=port,
        sslmode='require'
    )

    return conn

def get_function_names():
    conn = get_db_connection()

    cur = conn.cursor()
    cur.execute("""
        SELECT 
            specific_catalog, specific_schema, specific_name, routine_name, routine_type
        FROM 
            "information_schema"."routines"
    """)

    all_rows = cur.fetchall()
    fns = {}
    for row in all_rows:
        fns[row[2].split('_')[-1]] = f'%s.%s' % (row[1], row[3])
    return fns

def get_pg_classes():
    # Indexes and tables OIDs
    conn = get_db_connection()

    cur = conn.cursor()
    cur.execute("""
        SELECT 
            oid, relname
        FROM 
            "pg_catalog"."pg_class"
    """)

    all_rows = cur.fetchall()
    clas = {}
    for row in all_rows:
        clas[str(row[0])] = row[1]
    return clas

def get_pg_descriptions():
    # Indexes and tables OIDs
    conn = get_db_connection()

    cur = conn.cursor()
    cur.execute("""
        SELECT 
            objoid, description
        FROM 
            "pg_catalog"."pg_description"
    """)

    all_rows = cur.fetchall()
    desc = {}
    for row in all_rows:
        desc[str(row[0])] = row[1]
    return desc

def build_rte_arr(graph):
    rtable = graph['PLANNEDSTMT_1']['attrs'][':rtable']
    rarr = {}
    i = 0
    for node in rtable[0]:
        if isinstance(node, list):
            (node, ) = node
            ref = graph[node]['attrs'][':eref'][0]
            rarr[str(i+1)] = graph[ref]['attrs'][':aliasname'][0]
            i += 1
    return rarr

def get_pg_query_text(query_id):
    if query_id == '0':
        return '0'
    # Indexes and tables OIDs
    conn = get_db_connection(database='azure_sys')

    cur = conn.cursor()
    cur.execute(f"""
        SELECT
            "query_text_id", "query_sql_text", "query_type"
        FROM
            "query_store"."query_texts_view"
        WHERE
            ("query_text_id" = '%s')
    """ % (query_id))

    all_rows = cur.fetchall()
    return next(iter(all_rows))[1]

def get_query_types():
    return {
        '0': 'unknown',
        '1': 'select',
        '2': 'update',
        '3': 'insert',
        '4': 'delete',
        '5': 'utility',
        '6': 'nothing'
    }

def key_resolver(in_key, in_value, rarr):
    if in_key == ':funcid':
        return fns[in_value[0]]
    elif in_key == ':indexid':
        return clas[in_value[0]]
    elif in_key in [':opno', ':opfuncid', ':opresulttype']:
        return desc[in_value[0]]
    elif in_key == ':scanrelid':
        return rarr[in_value[0]]
    elif in_key == ':queryId':
        return get_pg_query_text(in_value[0])
    return in_value

def clean_graph(graph):
    rarr = build_rte_arr(graph)

    for node_key in graph.keys():
        graph[node_key]['attrs'] = {k: key_resolver(k, v, rarr) for k, v in graph[node_key]['attrs'].items()}

    return graph

def plot_graph_to_html(graph, graph_index):
    hide_keys = set([
        ':targetlist',
        ':parallel_aware',
        ':parallel_safe',
        ':plan_node_id',
        ':initPlan'
    ])

    hide_parent_keys = set([
        ':rtable',
        ':relationOids',
        ':subplans',
    ])

    def hide_key(node_name, in_key):
        if in_key in hide_keys:
            return False
        if 'PLANNEDSTMT' in node_name and in_key in hide_parent_keys:
            return False
        return True


    child2parent = {}
    labels = [] # Label for each node
    texts = [] # texts to be on the card
    values = [] # Value to use for sizing each node
    parents = [] # Parent for each node

    for k, v in graph.items():
        labels.append(k)
        texts.append('<br>'.join([str(k2) + ': ' + str([v2, v2[0]][isinstance(v2, list)]) for k2, v2 in v['attrs'].items() if hide_key(k, k2)]))
        values.append(v['cost'])

        for x in [child for child in v['edges'] if k != child]:
            child2parent[x] = k

    parents = [child2parent.get(x, '') for x, _ in graph.items()]


    fig = go.Figure(go.Icicle(
        labels=labels,
        text=texts,
        parents=parents,
        values=values,
        branchvalues="total",
        tiling=dict(
            orientation='v',
            flip='y'
        ),
        marker_colorscale='Reds',
        maxdepth=-1
    ))

    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))

    html_path = f'./static/html/%s.html' % (graph_index,)
    fig.write_html(html_path, include_plotlyjs=False)
    
    # Add plotly CDN to the html file
    with open(html_path, 'r') as file:
        filedata = file.read()

    # Replace the target string
    filedata = filedata.replace(
        '<head><meta charset="utf-8" /></head>',
        '<head><meta charset="utf-8" /><script src="https://cdn.plot.ly/plotly-2.12.1.min.js" id="plotlyjs"></script></head>'
    )

    # Write the file out again
    with open(html_path, 'w') as file:
        file.write(filedata)


def read_logs():
    query_data = []
    for i, filename in enumerate(os.listdir('./logs')):
        log_lines = []
        with open(os.path.join('./logs', filename), 'r') as f:
            for line in f.readlines():
                if line[0] == '2':
                    log_lines.append([])
                log_lines[-1].append(line)

        exc_plans = []
        for lines in log_lines:
            if 'PLANNEDSTMT' in lines[0]:
                lines[0] = lines[0].split('-DETAIL:')[-1].strip()
                exc_plans.append(''.join(lines))

        for j, plan in enumerate(tqdm(exc_plans)):
            eid = f'%s_%s' % (i, j)
            with open(f'./static/plans/%s.txt' % eid, 'w') as f2:
                f2.write(plan)
            graph = build_graph(log_string=plan)
            graph = clean_graph(graph)
            cost = graph['PLANNEDSTMT_1']['cost']
            query_data.append([
                i, j, cost,
                graph['PLANNEDSTMT_1']['attrs'][':queryId'],
                qts.get(graph['PLANNEDSTMT_1']['attrs'][':commandType'][0][0], 'OOV')
            ])
            tqdm.write('%s \t %s \t %s \t %s ' % (i, j, cost, graph['PLANNEDSTMT_1']['attrs'][':queryId'].replace('"', '\\"')))
            plot_graph_to_html(graph, eid)

    with open('query_data.pickle', 'wb') as handle:
        pickle.dump(query_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


fns = get_function_names()
clas = get_pg_classes()
desc = get_pg_descriptions()
qts = get_query_types()

read_logs()
