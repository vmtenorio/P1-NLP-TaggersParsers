def print_results(results, title):
    print(title)
    print("".join(['=']*len(title)))
    print("")
    print("Metric     | Precision |    Recall |  F1 Score | AligndAcc")
    print("-----------+-----------+-----------+-----------+-----------")
    for metric in["Tokens", "Sentences", "Words", "UPOS", "XPOS", "UFeats", "AllTags", "Lemmas", "UAS", "LAS", "CLAS", "MLAS", "BLEX"]:
        print("{:11}|{:10.2f} |{:10.2f} |{:10.2f} |{}".format(
            metric,
            100 * results[metric].precision,
            100 * results[metric].recall,
            100 * results[metric].f1,
            "{:10.2f}".format(100 * results[metric].aligned_accuracy) if results[metric].aligned_accuracy is not None else ""
        ))


def conll_text_reader(file):
    result = []
    sentence = []
    for line in file.readlines():
        if line == "\n":
            result.append(sentence)
            sentence = []
            continue
        if line.startswith('#'):
            continue
        toks = line.split('\t')
        if '-' not in toks[0]:
            sentence.append(toks[1])
    return result
