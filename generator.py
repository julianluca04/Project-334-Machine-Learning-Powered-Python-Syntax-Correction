import csv
import random

def generate_dataset(num_entries=10000, identity_ratio=0.1):
    templates = [
        ("print('{msg}')", ["prin('{msg}')", "primt('{msg}')", "print('{msg}'", "prant('{msg}')"]),
        ("if {var} == {val}:", ["if {var} = {val}:", "if {var} == {val}", "if {var} === {val}:"]),
        ("for {i} in range({val}):", ["fop {i} in range({val}):", "for {i} range({val}):", "for {i} in range({val})"]),
        ("while {var} < {val}:", ["while {var} < {val}", "whil {var} < {val}:", "while {var} {val}:"]),
        ("{var} += 1", ["{var} =+ 1", "{var} + 1", "{var} plus= 1"]),
        ("def {func}({arg}):", ["def {func}({arg})", "df {func}({arg}):", "def {func}{arg}:"]),
        ("return {var}", ["retun {var}", "retrun {var}", "return({var})"]),
        ("my_list = [{val}, {val2}, {val3}]", ["my_list = [{val} {val2} {val3}]", "my_list = ({val}, {val2}, {val3}]"])
    ]

    variables = ['x', 'y', 'i', 'n', 'count', 'data', 'val', 'item']
    messages = ['Hi', 'Hello', 'Error', 'Done', 'Start']
    functions = ['main', 'solve', 'check', 'parse', 'run']

    with open('synthetic_code_data.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['buggy_code', 'fixed_code'])
        writer.writeheader()

        for i in range(num_entries):
            # Fill out a template
            fixed_template, bugs = random.choice(templates)
            v, v2, v3 = random.sample(variables, 3)
            val, val2, val3 = random.randint(0, 10), random.randint(11, 20), random.randint(21, 30)
            m = random.choice(messages)
            fn = random.choice(functions)
            a = random.choice(['', 'x', 'a, b'])

            fixed = fixed_template.format(var=v, var2=v2, var3=v3, val=val, val2=val2, val3=val3, msg=m, i=v, func=fn, arg=a)
            
            # Decision: Create a bug or keep it as identity mapping?
            if random.random() < identity_ratio:
                buggy = fixed # Identity: No change needed
            else:
                buggy_pattern = random.choice(bugs)
                buggy = buggy_pattern.format(var=v, var2=v2, var3=v3, val=val, val2=val2, val3=val3, msg=m, i=v, func=fn, arg=a)

            writer.writerow({'buggy_code': buggy, 'fixed_code': fixed})

    print(f"Successfully generated {num_entries} rows (approx {int(num_entries*identity_ratio)} are identity mappings).")

if __name__ == "__main__":
    generate_dataset(10000)