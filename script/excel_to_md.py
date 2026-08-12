import csv
from collections import defaultdict
from pathlib import Path


REQUIRED_COLUMNS = {
    'Title', 'Short', 'Category', 'Level', 'Year', 'Publish', 'Paper',
    'Code', 'Website', 'Dataset', 'Input', 'Abstract'
}


def generate_markdown(csv_file, output_file):
    with open(csv_file, newline='', encoding='utf-8-sig') as source:
        reader = csv.DictReader(source)
        missing_columns = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing_columns:
            missing = ', '.join(sorted(missing_columns))
            raise ValueError(f'Missing required CSV columns: {missing}')
        paper_list = list(reader)

    category_dict = defaultdict(list)
    for paper_item in paper_list:
        category = paper_item['Category']
        if not category:
            raise ValueError(f"Paper has no category: {paper_item['Title']}")
        category_dict[category].append(paper_item)
    
    # Open the output markdown file
    with open(output_file, 'w', encoding='utf-8') as md_file:

        # Generate catelog with links
        md_file.write("## Table of contents\n\n")
        category_dict_keys = set(category_dict)
        category_dict_keys.remove('Survey')
        category_dict_keys = ['Survey'] + sorted(category_dict_keys)

        for category in category_dict_keys:
            md_file.write(f"- [{category}](#{category.replace(' ', '-').lower()})\n")
        md_file.write("\n")
        
        for category in category_dict_keys:
            md_file.write(f"## {category}\n\n")
            # Sort the paper list by Year
            category_dict[category].sort(
                key=lambda item: int(item['Year'].strip()), reverse=True
            )
            for idx, paper_item in enumerate(category_dict[category]):
                title = paper_item['Title']
                short = paper_item['Short']
                level = paper_item['Level']
                publish = paper_item['Publish']
                paper = paper_item['Paper']
                code = paper_item['Code']
                website = paper_item['Website']
                dataset = paper_item['Dataset']
                input_data = paper_item['Input']
                abstract = paper_item['Abstract']
                # Writing to the markdown file
                line_0 = f"### {idx+1}. {title}\n"
                md_file.write(line_0)
                line_00 = f"*{short}, {publish}*\n\n"
                md_file.write(line_00)

                line_1 = f"[📄 Paper]({paper})"
                if website:
                    line_1+=f" | [🌐 Project Page]({website})"
                if code:
                    line_1+=f" | [💻 Code]({code})"
                line_1+="\n"
                md_file.write(line_1)
                if level:
                    md_file.write(f"- Level: {level}\n")
                if dataset:
                    md_file.write(f"- Dataset: {dataset}\n")
                if input_data:
                    md_file.write(f"- Input: {input_data}\n")
                if abstract:
                    md_file.write("<details span>\n<summary><b>Abstract</b></summary>\n<br>\n\n")
                    md_file.write(f"{abstract}\n")
                    md_file.write("</details>\n\n")


def update_readme(readme_file, generated_file):
    readme = readme_file.read_text(encoding='utf-8')
    generated = generated_file.read_text(encoding='utf-8').rstrip() + '\n\n'
    table_marker = '## Table of contents\n'
    credits_marker = '## Credits\n'

    if table_marker not in readme or credits_marker not in readme:
        raise ValueError('README is missing the table-of-contents or credits marker')

    prefix = readme.split(table_marker, 1)[0]
    credits = credits_marker + readme.split(credits_marker, 1)[1]
    readme_file.write_text(prefix + generated + credits, encoding='utf-8')


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    generated_file = repo_root / 'output.md'
    generate_markdown(
        repo_root / 'excel' / 'paper_list.csv', generated_file
    )
    update_readme(repo_root / 'README.md', generated_file)
