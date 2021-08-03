content = []
for i in range(1, 1001):
    txt_path = f'result/res_img_{i}.txt'
    preds = [[str(int(p)) for p in line.strip().split(',')] for line in open(txt_path).readlines()]
    for pred in preds:
        content.append(f'{i},' + ','.join(pred) + ',1.0')
open('submit.csv', 'w').write('\n'.join(content))
