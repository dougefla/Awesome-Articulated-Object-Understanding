import pandas as pd

def generate_markdown(excel_file, output_file, lines_range=None):
    # Read the Excel file
    df = pd.read_excel(excel_file)

    # Open the output markdown file
    with open(output_file, 'w', encoding='utf-8') as md_file:
        for index, row in df.iterrows():
            # Extracting necessary information from each row
            title = row['Title']
            if "Act the Part" in title:
                pass
            short = row['Short']
            year = row['Year']
            publish = row['Publish']
            paper = row['Paper']
            website = row['Website']
            code = row['Code']
            task = row['Task']
            method = row['Method']
            network = row['Network Keywords']
            stages = row['Stages']
            dataset = row['Dataset']
            input = row['Input']
            abstract = row['Abstract']

            # Writing to the markdown file
            line_0 = f"### {title} [{publish}, {short}]\n"
            md_file.write(line_0)

            line_1 = f"[📄 Paper]({paper})"
            if not pd.isna(website):
                line_1+=f" | [🌐 Project Page]({website})"
            if not pd.isna(code):
                line_1+=f" | [💻 Code]({code})"
            line_1+="\n"
            md_file.write(line_1)

            if not pd.isna(task):
                md_file.write(f"- Task: {task}\n")
            if not pd.isna(method):
                md_file.write(f"- Method: {method}\n")
            if not pd.isna(network):
                md_file.write(f"- Network Keywords: {network}\n")
            if not pd.isna(stages):
                md_file.write(f"- Stages: {stages}\n")
            if not pd.isna(dataset):
                md_file.write(f"- Dataset: {dataset}\n")
            if not pd.isna(input):
                md_file.write(f"- Input: {input}\n")
            if abstract != "nan":
                md_file.write("<details open>\n<summary><b>Abstract</b></summary>\n<br>\n\n")
                md_file.write(f"{abstract}\n")
                md_file.write("</details>\n\n")

if __name__ == "__main__":
    generate_markdown('./excel/paper_list.xlsx', 'output.md')