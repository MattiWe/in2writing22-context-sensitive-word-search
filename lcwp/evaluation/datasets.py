import json
from collections import Counter
from statistics import mean
from pathlib import Path

import plotly.graph_objects as go


def dataset_statistics(output_path, input_file):
    """
    - number of queries
    - average number of results pre relevance score
    - number of POS classes

    :param output_path: Where to write the evaluation results
    :param input_file: Where to write the evaluation results
    """

    def _save_checkpoint(c, m, ln):
        result = {}

        for k, v in c.items():
            result[k] = dict(Counter(v))

        for k, v in m.items():
            result[k] = mean(v)

        print(result)

        open(output_path / f"dsstats-{input_file.stem}-{ln}.json", 'w').write(json.dumps(result))

    print(input_file)
    skip = {'query', 'id', 'answer', 'options'}
    counters = {}
    means = {}
    line_num = 0
    for line in open(input_file, 'r'):
        line = json.loads(line)
        line_num += 1
        for k, v in line.items():
            if k in skip:
                continue
            if isinstance(v, list):
                v = len(v)
                means.setdefault(k, []).append(v)
            counters.setdefault(k, []).append(v)
        if line_num % 10000000 == 0:
            _save_checkpoint(counters, means, line_num)
            counters = {}
            means = {}

    _save_checkpoint(counters, means, 'last')


def make_example_table(infiles: list, inqids: list):
    files = []
    res = {}
    indquis_done = set()
    for inf in infiles:
        files.append(inf.stem)
        for line in open(inf, 'r'):
            line = json.loads(line)
            if line['id'] in inqids:
                res.setdefault(line['id'], []).append(line['rel1'])
                indquis_done.add(line['id'])
                print(indquis_done, set(inqids))
                if indquis_done == set(inqids):
                    indquis_done = set()
                    break
    # qid - [[], [], [], [], [] ]
    for qid, rankings in res.items():
        for i in range(len(rankings[0])):
            print(f"{qid} & {' & '.join([r[i] for r in rankings])} \\\\")


def make_results_table(infiles: list):
    results = {inf.stem: json.loads(open(inf, 'r').read()) for inf in infiles}
    print(
        f"\\tt Netspeak & {results['wikitext-test-qlen3']['netspeak_results']['mrr']['shared_query_mrr']} & -- & {results['wikitext-test-qlen5']['netspeak_results']['mrr']['shared_query_mrr']} & -- & {results['cloth-test-qlen3']['netspeak_results']['mrr']['shared_query_mrr']} & -- & {results['cloth-test-qlen5']['netspeak_results']['mrr']['shared_query_mrr']} & \\\\")
    print(
        f"\\tt dBERT & {round(results['wikitext-test-qlen3']['model_results']['wikitext-test-mlm']['mrr']['shared_query_mrr'], 2)} & {round(results['wikitext-test-qlen3']['model_results']['wikitext-test-mlm']['mrr']['all'], 2)}"
        f" & {round(results['wikitext-test-qlen5']['model_results']['wikitext-test-mlm']['mrr']['shared_query_mrr'], 2)} & {round(results['wikitext-test-qlen5']['model_results']['wikitext-test-mlm']['mrr']['all'], 2)}"
        f" & {round(results['cloth-test-qlen3']['model_results']['cloth-test-mlm']['mrr']['shared_query_mrr'], 2)} & {round(results['cloth-test-qlen3']['model_results']['cloth-test-mlm']['mrr']['all'], 2)}"
        f" & {round(results['cloth-test-qlen5']['model_results']['cloth-test-mlm']['mrr']['shared_query_mrr'], 2)} & {round(results['cloth-test-qlen5']['model_results']['cloth-test-mlm']['mrr']['all'], 2)}"
        f" & {round(results['wikitext-test-qlen3']['model_results']['wikitext-test-mlm']['ndcg'][9], 2)} & {round(results['wikitext-test-qlen5']['model_results']['wikitext-test-mlm']['ndcg'][9], 2)}"
        f" & {round(results['cloth-test-qlen3']['model_results']['cloth-test-mlm']['ndcg'][9], 2)} & {round(results['cloth-test-qlen5']['model_results']['cloth-test-mlm']['ndcg'][9], 2)}"
        f" & {round(results['wikitext-test-qlen3']['model_results']['wikitext-test-mlm']['ndcg'][5], 2)} & {round(results['wikitext-test-qlen5']['model_results']['wikitext-test-mlm']['ndcg'][5], 2)}"
        f" & {round(results['cloth-test-qlen3']['model_results']['cloth-test-mlm']['ndcg'][5], 2)} & {round(results['cloth-test-qlen5']['model_results']['cloth-test-mlm']['ndcg'][5], 2)}"
        f"& \\\\")
    print(
        f"\\tt dBERT/Wt & {round(results['wikitext-test-qlen3']['model_results']['wikitext-test-mlm-finetuned']['mrr']['shared_query_mrr'], 2)} & {round(results['wikitext-test-qlen3']['model_results']['wikitext-test-mlm-finetuned']['mrr']['all'], 2)}"
        f" & {round(results['wikitext-test-qlen5']['model_results']['wikitext-test-mlm-finetuned']['mrr']['shared_query_mrr'], 2)} & {round(results['wikitext-test-qlen5']['model_results']['wikitext-test-mlm-finetuned']['mrr']['all'], 2)}"
        f" & {round(results['cloth-test-qlen3']['model_results']['cloth-test-mlm-finetuned']['mrr']['shared_query_mrr'], 2)} & {round(results['cloth-test-qlen3']['model_results']['cloth-test-mlm-finetuned']['mrr']['all'], 2)}"
        f" & {round(results['cloth-test-qlen5']['model_results']['cloth-test-mlm-finetuned']['mrr']['shared_query_mrr'], 2)} & {round(results['cloth-test-qlen5']['model_results']['cloth-test-mlm-finetuned']['mrr']['all'], 2)}"
        f" & {round(results['wikitext-test-qlen3']['model_results']['wikitext-test-mlm-finetuned']['ndcg'][9], 2)} & {round(results['wikitext-test-qlen5']['model_results']['wikitext-test-mlm-finetuned']['ndcg'][9], 2)}"
        f" & {round(results['cloth-test-qlen3']['model_results']['cloth-test-mlm-finetuned']['ndcg'][9], 2)} & {round(results['cloth-test-qlen5']['model_results']['cloth-test-mlm-finetuned']['ndcg'][9], 2)}"
        f" & {round(results['wikitext-test-qlen3']['model_results']['wikitext-test-mlm-finetuned']['ndcg'][5], 2)} & {round(results['wikitext-test-qlen5']['model_results']['wikitext-test-mlm-finetuned']['ndcg'][5], 2)}"
        f" & {round(results['cloth-test-qlen3']['model_results']['cloth-test-mlm-finetuned']['ndcg'][5], 2)} & {round(results['cloth-test-qlen5']['model_results']['cloth-test-mlm-finetuned']['ndcg'][5], 2)}"
        f"& \\\\")
    print(
        f"\\tt BART & {round(results['wikitext-test-qlen3']['model_results']['wikitext-test-bart']['mrr']['shared_query_mrr'], 2)} & {round(results['wikitext-test-qlen3']['model_results']['wikitext-test-bart']['mrr']['all'], 2)}"
        f" & {round(results['wikitext-test-qlen5']['model_results']['wikitext-test-bart']['mrr']['shared_query_mrr'], 2)} & {round(results['wikitext-test-qlen5']['model_results']['wikitext-test-bart']['mrr']['all'], 2)}"
        f" & {round(results['cloth-test-qlen3']['model_results']['cloth-test-bart']['mrr']['shared_query_mrr'], 2)} & {round(results['cloth-test-qlen3']['model_results']['cloth-test-bart']['mrr']['all'], 2)}"
        f" & {round(results['cloth-test-qlen5']['model_results']['cloth-test-bart']['mrr']['shared_query_mrr'], 2)} & {round(results['cloth-test-qlen5']['model_results']['cloth-test-bart']['mrr']['all'], 2)}"
        f" & {round(results['wikitext-test-qlen3']['model_results']['wikitext-test-bart']['ndcg'][9], 2)} & {round(results['wikitext-test-qlen5']['model_results']['wikitext-test-bart']['ndcg'][9], 2)}"
        f" & {round(results['cloth-test-qlen3']['model_results']['cloth-test-bart']['ndcg'][9], 2)} & {round(results['cloth-test-qlen5']['model_results']['cloth-test-bart']['ndcg'][9], 2)}"
        f" & {round(results['wikitext-test-qlen3']['model_results']['wikitext-test-bart']['ndcg'][5], 2)} & {round(results['wikitext-test-qlen5']['model_results']['wikitext-test-bart']['ndcg'][5], 2)}"
        f" & {round(results['cloth-test-qlen3']['model_results']['cloth-test-bart']['ndcg'][5], 2)} & {round(results['cloth-test-qlen5']['model_results']['cloth-test-bart']['ndcg'][5], 2)}"
        f"& \\\\")
    print(
        f"\\tt BART & {round(results['wikitext-test-qlen3']['model_results']['wikitext-test-bart-finetuned']['mrr']['shared_query_mrr'], 2)} & {round(results['wikitext-test-qlen3']['model_results']['wikitext-test-bart-finetuned']['mrr']['all'], 2)}"
        f" & {round(results['wikitext-test-qlen5']['model_results']['wikitext-test-bart-finetuned']['mrr']['shared_query_mrr'], 2)} & {round(results['wikitext-test-qlen5']['model_results']['wikitext-test-bart-finetuned']['mrr']['all'], 2)}"
        f" & {round(results['cloth-test-qlen3']['model_results']['cloth-test-bart-finetuned']['mrr']['shared_query_mrr'], 2)} & {round(results['cloth-test-qlen3']['model_results']['cloth-test-bart-finetuned']['mrr']['all'], 2)}"
        f" & {round(results['cloth-test-qlen5']['model_results']['cloth-test-bart-finetuned']['mrr']['shared_query_mrr'], 2)} & {round(results['cloth-test-qlen5']['model_results']['cloth-test-bart-finetuned']['mrr']['all'], 2)}"
        f" & {round(results['wikitext-test-qlen3']['model_results']['wikitext-test-bart-finetuned']['ndcg'][9], 2)} & {round(results['wikitext-test-qlen5']['model_results']['wikitext-test-bart-finetuned']['ndcg'][9], 2)}"
        f" & {round(results['cloth-test-qlen3']['model_results']['cloth-test-bart-finetuned']['ndcg'][9], 2)} & {round(results['cloth-test-qlen5']['model_results']['cloth-test-bart-finetuned']['ndcg'][9], 2)}"
        f" & {round(results['wikitext-test-qlen3']['model_results']['wikitext-test-bart-finetuned']['ndcg'][5], 2)} & {round(results['wikitext-test-qlen5']['model_results']['wikitext-test-bart-finetuned']['ndcg'][5], 2)}"
        f" & {round(results['cloth-test-qlen3']['model_results']['cloth-test-bart-finetuned']['ndcg'][5], 2)} & {round(results['cloth-test-qlen5']['model_results']['cloth-test-bart-finetuned']['ndcg'][5], 2)}"
        f"& \\\\")


def make_mrr_table(infiles: list):
    results = {inf.stem: json.loads(open(inf, 'r').read()) for inf in infiles}
    print(
        f"\\tt Netspeak & {results['wikitext-test-qlen3']['netspeak_results']['mrr']['shared_query_mrr']} & -- & {results['wikitext-test-qlen5']['netspeak_results']['mrr']['shared_query_mrr']} & -- & {results['cloth-test-qlen3']['netspeak_results']['mrr']['shared_query_mrr']} & -- & {results['cloth-test-qlen5']['netspeak_results']['mrr']['shared_query_mrr']} & \\\\")
    print(
        f"\\tt dBERT & {round(results['wikitext-test-qlen3']['model_results']['wikitext-test-mlm']['mrr']['shared_query_mrr'], 2)} & {round(results['wikitext-test-qlen3']['model_results']['wikitext-test-mlm']['mrr']['all'], 2)}"
        f" & {round(results['wikitext-test-qlen5']['model_results']['wikitext-test-mlm']['mrr']['shared_query_mrr'], 2)} & {round(results['wikitext-test-qlen5']['model_results']['wikitext-test-mlm']['mrr']['all'], 2)}"
        f" & {round(results['cloth-test-qlen3']['model_results']['cloth-test-mlm']['mrr']['shared_query_mrr'], 2)} & {round(results['cloth-test-qlen3']['model_results']['cloth-test-mlm']['mrr']['all'], 2)}"
        f" & {round(results['cloth-test-qlen5']['model_results']['cloth-test-mlm']['mrr']['shared_query_mrr'], 2)} & {round(results['cloth-test-qlen5']['model_results']['cloth-test-mlm']['mrr']['all'], 2)}"
        f"& \\\\")
    print(
        f"\\tt dBERT/Wt & {round(results['wikitext-test-qlen3']['model_results']['wikitext-test-mlm-finetuned']['mrr']['shared_query_mrr'], 2)} & {round(results['wikitext-test-qlen3']['model_results']['wikitext-test-mlm-finetuned']['mrr']['all'], 2)}"
        f" & {round(results['wikitext-test-qlen5']['model_results']['wikitext-test-mlm-finetuned']['mrr']['shared_query_mrr'], 2)} & {round(results['wikitext-test-qlen5']['model_results']['wikitext-test-mlm-finetuned']['mrr']['all'], 2)}"
        f" & {round(results['cloth-test-qlen3']['model_results']['cloth-test-mlm-finetuned']['mrr']['shared_query_mrr'], 2)} & {round(results['cloth-test-qlen3']['model_results']['cloth-test-mlm-finetuned']['mrr']['all'], 2)}"
        f" & {round(results['cloth-test-qlen5']['model_results']['cloth-test-mlm-finetuned']['mrr']['shared_query_mrr'], 2)} & {round(results['cloth-test-qlen5']['model_results']['cloth-test-mlm-finetuned']['mrr']['all'], 2)}"
        f"& \\\\")
    print(
        f"\\tt BART & {round(results['wikitext-test-qlen3']['model_results']['wikitext-test-bart']['mrr']['shared_query_mrr'], 2)} & {round(results['wikitext-test-qlen3']['model_results']['wikitext-test-bart']['mrr']['all'], 2)}"
        f" & {round(results['wikitext-test-qlen5']['model_results']['wikitext-test-bart']['mrr']['shared_query_mrr'], 2)} & {round(results['wikitext-test-qlen5']['model_results']['wikitext-test-bart']['mrr']['all'], 2)}"
        f" & {round(results['cloth-test-qlen3']['model_results']['cloth-test-bart']['mrr']['shared_query_mrr'], 2)} & {round(results['cloth-test-qlen3']['model_results']['cloth-test-bart']['mrr']['all'], 2)}"
        f" & {round(results['cloth-test-qlen5']['model_results']['cloth-test-bart']['mrr']['shared_query_mrr'], 2)} & {round(results['cloth-test-qlen5']['model_results']['cloth-test-bart']['mrr']['all'], 2)}"
        f"& \\\\")
    print(
        f"\\tt BART/WN & {round(results['wikitext-test-qlen3']['model_results']['wikitext-test-bart-finetuned']['mrr']['shared_query_mrr'], 2)} & {round(results['wikitext-test-qlen3']['model_results']['wikitext-test-bart-finetuned']['mrr']['all'], 2)}"
        f" & {round(results['wikitext-test-qlen5']['model_results']['wikitext-test-bart-finetuned']['mrr']['shared_query_mrr'], 2)} & {round(results['wikitext-test-qlen5']['model_results']['wikitext-test-bart-finetuned']['mrr']['all'], 2)}"
        f" & {round(results['cloth-test-qlen3']['model_results']['cloth-test-bart-finetuned']['mrr']['shared_query_mrr'], 2)} & {round(results['cloth-test-qlen3']['model_results']['cloth-test-bart-finetuned']['mrr']['all'], 2)}"
        f" & {round(results['cloth-test-qlen5']['model_results']['cloth-test-bart-finetuned']['mrr']['shared_query_mrr'], 2)} & {round(results['cloth-test-qlen5']['model_results']['cloth-test-bart-finetuned']['mrr']['all'], 2)}"
        f"& \\\\")
    print(
        f"\\tt BERT/WN & {round(results['wikitext-test-qlen3']['model_results']['wikitext-test-mlm-ft-netspeak']['mrr']['shared_query_mrr'], 2)} & {round(results['wikitext-test-qlen3']['model_results']['wikitext-test-mlm-ft-netspeak']['mrr']['all'], 2)}"
        f" & {round(results['wikitext-test-qlen5']['model_results']['wikitext-test-mlm-ft-netspeak']['mrr']['shared_query_mrr'], 2)} & {round(results['wikitext-test-qlen5']['model_results']['wikitext-test-mlm-ft-netspeak']['mrr']['all'], 2)}"
        f" & {round(results['cloth-test-qlen3']['model_results']['cloth-test-mlm-ft-netspeak']['mrr']['shared_query_mrr'], 2)} & {round(results['cloth-test-qlen3']['model_results']['cloth-test-mlm-ft-netspeak']['mrr']['all'], 2)}"
        f" & {round(results['cloth-test-qlen5']['model_results']['cloth-test-mlm-ft-netspeak']['mrr']['shared_query_mrr'], 2)} & {round(results['cloth-test-qlen5']['model_results']['cloth-test-mlm-ft-netspeak']['mrr']['all'], 2)}"
        f"& \\\\")


def _model_name(m):
    return "-".join(m.split('-')[2:])


def make_ndgc_plots(infiles: list):
    results = {inf.stem: json.loads(open(inf, 'r').read()) for inf in infiles}
    fig = go.Figure()
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
         11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
         21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    colors = {
        'qlen3': {
            'mlm': 'rgb(255,133,13)',
            'mlm-finetuned': 'rgb(127,195,28)',
            'mlm-ft-netspeak': 'rgb(0,195,28)',
            'bart': 'rgb(11,68,153)',
            'bart-finetuned':'rgb(238,1,76)'
        },
        'qlen5':{
            'mlm': 'rgb(239,167,108)',
            'mlm-finetuned': 'rgb(166,167,111)',
            'mlm-ft-netspeak': 'rgb(0,167,111)',
            'bart': 'rgb(40,92,153)',
            'bart-finetuned': 'rgb(226,71,127)'
        }
    }

    for ds_id, result in results.items():
        for mkey, model_results in result['model_results'].items():
            y = model_results['ndcg'][0:5] + [None, None, None, None] + model_results['ndcg'][5:6] + \
                [None, None, None, None] + model_results['ndcg'][6:7] + [None, None, None, None] + \
                model_results['ndcg'][7:8] + [None, None, None, None] + model_results['ndcg'][8:9] + \
                [None, None, None, None] + model_results['ndcg'][9:10]
            model = '-'.join(mkey.split("-")[2:])
            qlen = ds_id.split("-")[-1]
            if ds_id.split("-")[0] == 'cloth':
                fig.add_trace(go.Scatter(x=x, y=y, name=f"{_model_name(mkey)} on {ds_id}", connectgaps=True,
                                         mode='lines+markers', line=dict(dash='dash', color=colors[qlen][model])))
            else:
                fig.add_trace(go.Scatter(x=x, y=y, name=f"{_model_name(mkey)} on {ds_id}", connectgaps=True,
                                         mode='lines+markers', line=dict(color=colors[qlen][model])))

    fig.update_layout(
        xaxis=dict(
            showline=True,
            showgrid=True,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=1,
            ticks='outside'
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=False,
            showline=True,
            linewidth=1,
            showticklabels=True,
        ),
        plot_bgcolor='white',
    )
    fig.write_image("ndcg-plot.pdf")


def make_pos_plots(infiles: list):
    results = {inf.stem: json.loads(open(inf, 'r').read()) for inf in infiles}
    pos_stats = {}

    for ds_id, result in results.items():
        for mkey, model_results in result['model_results'].items():
            for posc_key, posc_value in model_results['mrr']['pos_class'].items():
                # {'NOUN': {gpt: [1, 2, 3, 4]}, }
                pos_stats.setdefault(posc_key, {}).setdefault('netspeak', []).append(result['netspeak_results']['mrr']['pos_class'][posc_key])
                pos_stats.setdefault(posc_key, {}).setdefault(_model_name(mkey), []).append(posc_value)

    pos_tags = []  # labels
    models = {}
    for posc, posv in pos_stats.items():
        pos_tags.append(posc)
        for mkey, mval in posv.items():
            models.setdefault(mkey, []).append(mean(mval))

    data = []
    for mkey in models:
        data.append(go.Bar(name=mkey, x=pos_tags, y=models[mkey]))
    fig = go.Figure(data=data)
    fig.update_layout(barmode='group', bargap=0.15, bargroupgap=0.1)
    fig.update_layout(
        xaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside'
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=False,
            showline=False,
            showticklabels=True,
            linewidth=2,
        ),
        plot_bgcolor='white'
    )
    fig.write_image("pos-plot.pdf")


def make_position_plots(infiles: list, suffix):
    results = {inf.stem: json.loads(open(inf, 'r').read()) for inf in infiles}
    pos_stats = {}

    for ds_id, result in results.items():
        for mkey, model_results in result['model_results'].items():
            for posc_key, posc_value in model_results['mrr']['position'].items():
                print(posc_key, mkey, posc_value)
                pos_stats.setdefault(posc_key, {}).setdefault(_model_name(mkey), []).append(posc_value)
                pos_stats.setdefault(posc_key, {}).setdefault('netspeak', []).append(result['netspeak_results']['mrr']['position'][posc_key])

    pos_tags = []  # labels
    models = {}
    for posc, posv in pos_stats.items():
        pos_tags.append(posc)
        for mkey, mval in posv.items():
            models.setdefault(mkey, []).append(mean(mval))

    data = []
    for mkey in models:
        data.append(go.Bar(name=mkey, x=pos_tags, y=models[mkey]))
    fig = go.Figure(data=data)
    fig.update_layout(barmode='group', bargap=0.15, bargroupgap=0.1)
    fig.update_layout(
        xaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=2,
            ticks='outside'
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=False,
            showline=False,
            showticklabels=True,
            linewidth=2,
        ),
        plot_bgcolor='white'
    )
    fig.write_image(f"position-plot-{suffix}.pdf")


if __name__ == "__main__":
    # np = Path("/mnt/ceph/storage/data-in-progress/data-research/netspeak/in2writing22-lcwp/training-datasets")
    op = Path("C:/Users/Leaves/Sync")
    # pp = Path("/mnt/ceph/storage/data-in-progress/data-research/netspeak/in2writing22-lcwp/predictions/fin")
    # for infile in np.glob("*.ndjson"):
    #     if infile.is_dir():
    #         continue
    #     dataset_statistics(op, infile)

    # print examples

    # make_example_table([pp / 'cloth-test-mlm.ndjson', pp / 'cloth-test-bart.ndjson',
    #                     pp / 'cloth-test-mlm-finetuned.ndjson', pp / 'wikitext-test-mlm.ndjson',
    #                     pp / 'wikitext-test-bart.ndjson', pp / 'wikitext-test-mlm-finetuned.ndjson'],
    #                    ['e5b810d9-d581-44c2-97fe-b2044966061d', 'd5235a63-0cba-4e87-9942-90904fb6f46c',
    #                     '53bdaf07-30ac-4900-8e43-8ec7629aaa42', '09ccda2c-cee0-430b-9656-e01af7aaf0a8',
    #                     '4055dd1f-1215-472d-95c0-b0b4008ca110', '507c40ee-65e8-4fa7-9a15-4456b940f5e0'])
    make_mrr_table([op / 'wikitext-test-qlen3.json', op / 'wikitext-test-qlen5.json',
                    op / 'cloth-test-qlen3.json', op / 'cloth-test-qlen5.json'])

    make_ndgc_plots([op / 'wikitext-test-qlen3.json', op / 'wikitext-test-qlen5.json',
                    op / 'cloth-test-qlen3.json', op / 'cloth-test-qlen5.json'])

    make_pos_plots([op / 'wikitext-test-qlen3.json', op / 'wikitext-test-qlen5.json',
                    op / 'cloth-test-qlen3.json', op / 'cloth-test-qlen5.json'])

    make_position_plots([op / 'wikitext-test-qlen3.json', op / 'cloth-test-qlen3.json'], 3)
    make_position_plots([op / 'wikitext-test-qlen5.json', op / 'cloth-test-qlen5.json'], 5)
