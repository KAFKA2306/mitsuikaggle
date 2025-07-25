import os
import sys

def read_all_project_files(base_dir):
    """
    プロジェクトディレクトリ内のすべてのファイルを読み込み、その内容を辞書として返します。
    特定のディレクトリやファイルをスキップします。
    """
    file_contents = {}
    exclude_dirs = ['input/kaggle_evaluation']
    
    for root, dirs, files in os.walk(base_dir):
        # 除外ディレクトリの処理
        # os.walkはdirsリストをin-placeで変更することで、特定のサブディレクトリへの再帰をスキップできる
        dirs[:] = [d for d in dirs if not any(os.path.join(root, d).startswith(os.path.join(base_dir, ed)) for ed in exclude_dirs)]

        for file in files:
            file_path = os.path.join(root, file)
            relative_file_path = os.path.relpath(file_path, base_dir)

            # :Zone.Identifier ファイルをスキップ
            if ":Zone.Identifier" in file:
                continue

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    file_contents[relative_file_path] = content
            except UnicodeDecodeError:
                # バイナリファイルをスキップ
                print(f"Skipping binary file: {relative_file_path}", file=sys.stderr)
            except Exception as e:
                print(f"Error reading {relative_file_path}: {e}", file=sys.stderr)
    return file_contents

if __name__ == "__main__":
    project_root = "/home/kafka/finance/mitsui-commodity-prediction-challenge"
    all_files_content = read_all_project_files(project_root)

    # 結果を整形して標準出力に表示
    formatted_output = ""
    for file_path, content in all_files_content.items():
        formatted_output += f"--- FILE: {file_path} ---\n"
        formatted_output += content
        formatted_output += "\n--- END FILE: {file_path} ---\n\n"
    
    print(formatted_output)