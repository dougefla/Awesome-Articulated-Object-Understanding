import pandas as pd

class PaperItem:
    def __init__(self, title, short, category, level, year, publish, paper, code, website, dataset, input, abstract):
        self.title = title
        self.short = short
        self.category = category
        self.level = level
        self.year = year
        self.publish = publish
        self.paper = paper
        self.code = code
        self.website = website
        self.dataset = dataset
        self.input = input
        self.abstract = abstract

def generate_markdown(excel_file, output_file, lines_range=None):
    # Read the Excel file
    df = pd.read_excel(excel_file)
    paper_list = []
    category_dict = {}
    for index, row in df.iterrows():
        paper_item = PaperItem(row['Title'], row['Short'], row['Category'], row['Level'], row['Year'], row['Publish'], row['Paper'], row['Code'], row['Website'], row['Dataset'], row['Input'], row['Abstract'])
        paper_list.append(paper_item)
        if row['Category'] not in category_dict.keys():
            category_dict[row['Category']] = []
        category_dict[row['Category']].append(paper_item)
    
    # Open the output markdown file
    with open(output_file, 'w', encoding='utf-8') as md_file:

        # Generate catelog with links
        md_file.write("## Table of contents\n\n")
        for category in sorted(category_dict.keys()):
            md_file.write(f"- [{category}](#{category.replace(' ', '-').lower()})\n")
        md_file.write("\n")
        
        for category in sorted(category_dict.keys()):
            md_file.write(f"## {category}\n\n")
            # Sort the paper list by Year
            category_dict[category].sort(key=lambda x: x.year, reverse=True)
            for idx, paper_item in enumerate(category_dict[category]):
                title = paper_item.title
                short = paper_item.short
                level = paper_item.level
                publish = paper_item.publish
                paper = paper_item.paper
                code = paper_item.code
                website = paper_item.website
                dataset = paper_item.dataset
                input = paper_item.input
                abstract = paper_item.abstract
                # Writing to the markdown file
                line_0 = f"### {idx+1}. {title}\n"
                md_file.write(line_0)
                line_00 = f"*{short}, {publish}*\n\n"
                md_file.write(line_00)

                line_1 = f"[📄 Paper]({paper})"
                if not pd.isna(website):
                    line_1+=f" | [🌐 Project Page]({website})"
                if not pd.isna(code):
                    line_1+=f" | [💻 Code]({code})"
                line_1+="\n"
                md_file.write(line_1)
                if not pd.isna(level):
                    md_file.write(f"- Level: {level}\n")
                if not pd.isna(dataset):
                    md_file.write(f"- Dataset: {dataset}\n")
                if not pd.isna(input):
                    md_file.write(f"- Input: {input}\n")
                if abstract != "nan":
                    md_file.write("<details span>\n<summary><b>Abstract</b></summary>\n<br>\n\n")
                    md_file.write(f"{abstract}\n")
                    md_file.write("</details>\n\n")

if __name__ == "__main__":
    generate_markdown('./excel/paper_list.xlsx', 'output.md')