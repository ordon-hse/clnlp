from ngram import NGram

def func():
    model = NGram(log=True)
    print(len(model.vocab), len(model.vocab)**2)
    model.train(smoothing='interpolation')
    # return
    # model.train(smoothing='add-k')
    test_sents_from_text = [
        'Нет, лучше сказать',
        'Генерал еще вчера обещал',
        'Воображаю, как Мари удивлялась',
        # 'Они ей рассказали, что',
        # 'князь рассказал всё в подробности',
        'я вас сейчас обниму и поцелую',
        # 'благодаря хорошей погоде уже распустились все деревья',
    ]

    print('from text:')
    for sent in test_sents_from_text:
        print(f'probability of "{sent}": {model.probability(sent)}')
    
    test_sents_not_from_text = [
        'Машина Тьюринга является расширением',
        'интуитивный алгоритм может быть реализован',
        'процесс пошагового вычисления, в котором',
        'можно вычислить всё, что можно',
    ]
    print('not from text:')
    for sent in test_sents_not_from_text:
        print(f'probability of "{sent}": {model.probability(sent)}')

func()
