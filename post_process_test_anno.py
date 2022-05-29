from pathlib import Path
import pickle

def main():
    anno_path = Path('result.pkl')
    with anno_path.open('rb') as f:
        annotations = pickle.load(f)
        print(annotations)
        print(len(annotations))
        file_count = len([fn for fn in Path('data/OBJ_Train_Datasets/Public_Image').iterdir()])
        print(file_count)

if __name__ == '__main__':
    main()