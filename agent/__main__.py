import os
import sys
import argparse
import logging.config
import yaml

import agent.config as config
from agent.trainer.session import SessionRunner, SessionMetricsPresenter


def main(args=None):
    check_python_version()
    setup_logging()
    logger = logging.getLogger(__name__)
    try:
        parse_command_line_arguments_and_launch()
    except:
        logger.error('Unexpected error:', exc_info=True)
        raise


def parse_command_line_arguments_and_launch():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=['train-new', 'train-resume', 'play', 'visualize-tsne', 'metrics-show', 'metrics-export'])
    parser.add_argument("-s", type=str, help="session id")
    parser.add_argument("--resultspath", help="root for training result sessions")
    parser.add_argument("--ec2spot", help="use this options if the trainer is executed in a AWS EC2 Spot Instance",
                        action="store_true")
    parsed_arguments = parser.parse_args()
    validate_arguments(parsed_arguments)
    if parsed_arguments.resultspath:
        config.train_results_root_folder = parsed_arguments.resultspath
    if parsed_arguments.ec2spot:
        config.trained_using_aws_spot_instance = True
    if parsed_arguments.action == 'train-new':
        SessionRunner(config).train_new()
    elif parsed_arguments.action == 'train-resume':
        SessionRunner(config).train_resume(session_id=parsed_arguments.s)
    elif parsed_arguments.action == 'play':
        SessionRunner(config).play(session_id=parsed_arguments.s)
    elif parsed_arguments.action == 'visualize-tsne':
        SessionRunner(config).play_and_visualize_q_network_tsne(session_id=parsed_arguments.s)
    elif parsed_arguments.action == 'metrics-show':
        SessionMetricsPresenter(config).show(session_id=parsed_arguments.s)
    elif parsed_arguments.action == 'metrics-export':
        SessionMetricsPresenter(config).save_to_image(session_id=parsed_arguments.s)


def validate_arguments(parsed_arguments):
    if (parsed_arguments.action == (
                'train-resume' or 'play' or 'visualize-tsne' or 'metrics-show' or 'metrics-export')) and not parsed_arguments.s:
        raise SystemExit("-s argument (session id) is required for action {0}".format(parsed_arguments.action))


def check_python_version():
    if sys.version_info.major != 2 or sys.version_info.minor < 7:
        raise SystemExit("Python version 2.7 required")


def setup_logging(path='config_logging.yaml', default_level=logging.INFO):
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        print("Logging configuration \"{0}\" was not found. Using default basic configuration instead".format(path))
        logging.basicConfig(level=default_level)


if __name__ == "__main__":
    main()