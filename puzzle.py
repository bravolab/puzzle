from gooey import Gooey, GooeyParser
from puzzlebox import PuzzleBox
import os


@Gooey(program_name="Puzzle", image_dir=os.path.abspath(os.getcwd() + '/img'))
def parse_args():
    parser = GooeyParser()
    parser.add_argument('--model',
                        metavar="Model",
                        required=True,
                        widget="Dropdown",
                        choices=["dt"],
                        default="dt")

    parser.add_argument("--action",
                        metavar="Action",
                        required=True,
                        widget="Dropdown",
                        choices=["train", "analyze"],
                        default="train")

    parser.add_argument('--input',
                        action='store',
                        widget='FileChooser',
                        metavar="Input Data")

    parser.add_argument('--target',
                        action='store',
                        widget='TextField',
                        metavar="Target Variable")

    parser.add_argument('--data',
                        action='store',
                        widget='FileChooser',
                        metavar="Test Data")

    parser.add_argument('--model_name',
                        action='store',
                        widget='TextField',
                        metavar="Model Name")

    parser.add_argument("--plot",
                        metavar="Choose a plot type",
                        required=True,
                        widget="Dropdown",
                        choices=["auc", "boundary"],
                        default="auc")

    parsed_args = parser.parse_args()
    return parsed_args


if __name__ == '__main__':
    args = parse_args()
    input_file = args.input
    model_name = args.model_name
    model = args.model
    target = args.target
    plot = args.plot
    data = args.data
    puzzle = PuzzleBox(input_file, model_name)

    model = puzzle.create_model(model, target)
    plot = puzzle.plot_results(model, plot)
    puzzle.evaluate(model)
    puzzle.save(model)
    model = puzzle.load()
    puzzle.predict(model, data)



