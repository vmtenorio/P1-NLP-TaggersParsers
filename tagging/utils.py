def tups_to_file(path, tuples):
    with open(path, 'w') as f:
        for t in tuples:
            f.write(str(t) + '\n')


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

