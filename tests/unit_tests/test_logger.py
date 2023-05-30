import os
import logging
from typing import Any
from logger import get_logger, log_error, close_handlers


def test_get_logger(caplog: Any) -> None:
    """
    Tests the `get_logger` function.

    This function tests the creation of a logger object by the `get_logger` function.
    It checks that the logger has the correct level, name, and handlers.
    It also checks that a log message is correctly captured and written to the log file.

    Args:
        caplog (Any): A pytest fixture that captures log output.

    Raises:
        AssertionError: If any of the assertions fail.
    """
    # Given
    log_file_path = "test_log.txt"
    task_name = "Test task"

    # When
    logger = get_logger(log_file_path, task_name)

    # Then
    assert logger.level == logging.INFO
    assert logger.name == task_name
    assert len(logger.handlers) == 2
    assert isinstance(logger.handlers[0], logging.StreamHandler)
    assert isinstance(logger.handlers[1], logging.FileHandler)

    # Log a message to test handlers
    logger.info("Test log message")

    assert "Test log message" in caplog.text

    # Check if the log file was created and has the log message
    with open(log_file_path, "r", encoding="utf-8") as log_file:
        log_content = log_file.read()

    assert "Test log message" in log_content

    # Close handlers
    close_handlers(logger)

    # Clean up
    os.remove(log_file_path)


def test_get_logger_reset_file() -> None:
    """
    Tests the `reset_file` functionality of `get_logger` function.

    This function tests that the `get_logger` function correctly resets a log
    file when `reset_file` is set to True.
    It first writes a log message to a log file, then calls `get_logger` with
    `reset_file=True`, and finally checks that the log file is empty.

    Raises:
        AssertionError: If any of the assertions fail.
    """
    # Given
    log_file_path = "test_log_reset.log"
    task_name = "Test task"

    # Create a log file with a log message
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        log_file.write("Initial log message")

    # When
    logger = get_logger(log_file_path, task_name, reset_file=True)

    # Then
    # Check if the log file is empty
    with open(log_file_path, "r", encoding="utf-8") as log_file:
        log_content = log_file.read()

    assert log_content == ""

    # Close handlers
    close_handlers(logger)

    # Clean up
    os.remove(log_file_path)


def test_get_logger_no_reset_file() -> None:
    """
    Tests the `reset_file` functionality of `get_logger` function.

    This function tests that the `get_logger` function correctly appends to a
    log file when `reset_file` is set to False.
    It first writes a log message to a log file, then calls `get_logger` with
    `reset_file=False`, writes another log message, and finally checks that the
    log file contains both log messages.

    Raises:
        AssertionError: If any of the assertions fail.
    """
    # Given
    log_file_path = "test_log_no_reset.log"
    task_name = "Test task"

    # Create a log file with a log message
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        log_file.write("Initial log message")

    # When
    logger = get_logger(log_file_path, task_name, reset_file=False)
    logger.info("Second log message")

    # Then
    # Check if the log file contains both log messages
    with open(log_file_path, "r", encoding="utf-8") as log_file:
        log_content = log_file.read()

    assert "Initial log message" in log_content
    assert "Second log message" in log_content

    # Close handlers
    close_handlers(logger)

    # Clean up
    os.remove(log_file_path)


def test_log_error(tmpdir: Any) -> None:
    """
    Tests the `log_error` function.

    This function tests the writing of an error message and traceback to an
    error file by the `log_error` function.
    It checks that the error message, exception, and traceback are correctly
    written to the error file.

    Args:
        tmpdir (Any): A pytest fixture that provides a temporary directory
        unique to the test invocation.

    Raises:
        AssertionError: If any of the assertions fail.
    """
    # Given
    message = "Test error message"
    error_file_path = tmpdir.join("error.log")

    # When
    try:
        raise Exception("Test Exception")
    except Exception as error:
        log_error(message, error, str(error_file_path))

    # Then
    with open(error_file_path, 'r', encoding="utf-8") as file:
        error_msg = file.read()

    assert message in error_msg
    assert "Test Exception" in error_msg
    assert "Traceback" in error_msg
