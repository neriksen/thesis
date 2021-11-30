import pandas as pd

# Create input with pbpaste > input_file.csv


def replace_first_dot(in_str: str):
    if in_str[:5].lower() == "joint":
        in_str = f'{in_str[:5]}.{in_str[5:]}'
    if len(in_str.split(".")) >= 3:
        return in_str.replace(".", "_", 1)
    return in_str


def main(file_name):
    input_root = "../../data/"
    input_file = f"{input_root}{file_name}.csv"
    df = pd.read_table(input_file, delim_whitespace=True)
    df.drop(columns='Pr(>|t|)', inplace=True)
    columns = ['Parameter', 'Estimate', 'Std_Error', 't-value', 'Pr(>|t|)']

    df.columns = columns
    df.set_index("Parameter", drop=True, inplace=True)
    df.index = df.index.str.replace("[", "")
    df.index = df.index.str.replace("]", "")
    df.index = df.index.map(replace_first_dot)


    print(df)
    df.to_csv(f"{input_root}{file_name}.csv")

if __name__ == '__main__':
    main("fit_sGARCH10")