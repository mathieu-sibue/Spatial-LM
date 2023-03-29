
# from LMs.Roberta import RobertaClassifier
from LMs.LayoutLM import LayoutLMTokenclassifier
from LMs.LayoutLM import LayoutLM4DocVQA
from LMs.Roberta import GraphRobertaTokenClassifier, RobertaTokenClassifier
from LMs.CSModel import CSTokenClassifier,CSMaskedLM, KeyValLinking

def setup(opt):
    print('network:' + opt.network_type)
    # if opt.network_type == 'roberta':
    #     model = RobertaClassifier(opt)
    if opt.network_type == 'layoutlm':
        if opt.task_type == 'token-classifier':
            model = LayoutLMTokenclassifier(opt)
        elif opt.task_type == 'docvqa':
            model = LayoutLM4DocVQA(opt)
    elif opt.network_type == 'graph_roberta':
        model = GraphRobertaTokenClassifier(opt)
    elif opt.network_type == 'roberta':
        model = RobertaTokenClassifier(opt)
    elif opt.network_type == 'csmodel':
        if opt.task_type == 'cspretrain':
            print('model: load masked-csmodel for mlm')
            model = CSMaskedLM(opt)
        elif opt.task_type == 'token-classifier':
            print('model: load csmodel for token classification')
            model = CSTokenClassifier(opt)
        elif opt.task_type == 'link-binary':
            model = KeyValLinking(opt)
    else:
        raise Exception('model not supported:{}'.format(opt.network_type))

    return model


