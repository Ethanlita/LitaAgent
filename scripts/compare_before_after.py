#!/usr/bin/env python
"""对比修改前后的表现"""

# 对比三次比赛
data = {
    '20260111_172250 (修改前)': {
        'score': 1.0296,
        'rank': 9,
        'buyer_shortfall': 24.7,
        'buyer_exact': 8.0,
        'buyer_overfull': 67.7,
        'buyer_overfill_need': 32.7,
    },
    '20260111_233038 (修改后)': {
        'score': 1.0051,
        'rank': 10,
        'buyer_shortfall': 22.4,
        'buyer_exact': 11.9,
        'buyer_overfull': 65.8,
        'buyer_overfill_need': 34.2,
    },
}

print('=== LOS 表现对比 ===')
print(f"{'指标':<25} {'修改前':<15} {'修改后':<15} {'变化':<10}")
print('-' * 65)

metrics = [
    ('Score', 'score'),
    ('排名', 'rank'),
    ('BUYER Shortfall%', 'buyer_shortfall'),
    ('BUYER Exact%', 'buyer_exact'),
    ('BUYER Overfull%', 'buyer_overfull'),
    ('BUYER Overfill/Need%', 'buyer_overfill_need'),
]

before = data['20260111_172250 (修改前)']
after = data['20260111_233038 (修改后)']

for name, key in metrics:
    b = before[key]
    a = after[key]
    diff = a - b
    sign = '+' if diff > 0 else ''
    good = ''
    if key == 'buyer_exact':
        good = '✓' if diff > 0 else '✗'
    elif key in ['buyer_shortfall', 'buyer_overfull', 'buyer_overfill_need']:
        good = '✓' if diff < 0 else '✗'
    elif key == 'score':
        good = '✓' if diff > 0 else '✗'
    elif key == 'rank':
        good = '✓' if diff < 0 else '✗'
    print(f'{name:<25} {b:<15} {a:<15} {sign}{diff:.1f} {good}')
