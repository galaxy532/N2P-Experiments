# Few-shot accuracy — llama3-8b (k=4)

| task | tier | framing | exact-value acc | first-token acc | single-tok answers |
|---|---|---|---|---|---|
| addition | clean-core | symbolic | 1.000 | 1.000 | 1.00 |
| addition | clean-core | word | 1.000 | 1.000 | 1.00 |
| addition | clean-core | wordproblem | 1.000 | 1.000 | 1.00 |
| subtraction | clean-core | symbolic | 1.000 | 1.000 | 0.50 |
| subtraction | clean-core | word | 1.000 | 1.000 | 0.50 |
| subtraction | clean-core | wordproblem | 1.000 | 1.000 | 0.50 |
| mult_const | clean-core | symbolic | 1.000 | 1.000 | 1.00 |
| mult_const | clean-core | word | 1.000 | 1.000 | 1.00 |
| mult_const | clean-core | wordproblem | 1.000 | 1.000 | 1.00 |
| multiplication | stress-set | symbolic | 1.000 | 1.000 | 1.00 |
| multiplication | stress-set | word | 1.000 | 1.000 | 1.00 |
| multiplication | stress-set | wordproblem | 1.000 | 1.000 | 1.00 |
| int_division | stress-set | symbolic | 0.980 | 0.980 | 1.00 |
| int_division | stress-set | word | 0.960 | 0.960 | 1.00 |
| int_division | stress-set | wordproblem | 0.950 | 0.950 | 1.00 |
| modular | stress-set | symbolic | 0.150 | 0.150 | 1.00 |
| modular | stress-set | word | 0.150 | 0.150 | 1.00 |
| modular | stress-set | wordproblem | 0.090 | 0.090 | 1.00 |
