"""Shared logging setup for the project.

Provides a helper to configure module loggers with a file handler
writing to `logs.log` in the project root. This keeps logging
configuration consistent across modules.
"""

import logging
import os
from typing import Optional


def configure_file_logger(name: Optional[str] = None, log_file: Optional[str] = None, level: int = logging.DEBUG) -> logging.Logger:
    """Return a logger configured with a FileHandler writing to `logs.log`.

    If a FileHandler for the same `log_file` is already attached to the
    logger, this function leaves it in place.

    Additionally, when used by modules that define a `RAGModel` or `Assistant`
    class (for example `rag_based_llm.py`), this function will attempt to
    wrap key methods on those classes with lightweight logging wrappers so
    that calls are logged without modifying the original module source.
    """
    if log_file is None:
        log_file = os.path.join(os.getcwd(), "logs.log")

    logger = logging.getLogger(name)
    # Ensure callers can safely call this function multiple times
    logger.addHandler(logging.NullHandler())

    # If a file handler for this exact file already exists, do nothing
    if any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == log_file
           for h in logger.handlers):
        return logger

    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        handler.setLevel(level)

        logger.addHandler(handler)
        logger.setLevel(level)
        logger.debug("Logging initialized to %s", log_file)
    except Exception:
        # Never let logging setup break the application
        pass

    # --- Runtime instrumentation for modules that import this logger ---
    try:
        import inspect
        import functools
        import sys

        # Inspect the call stack to find the importing module's globals
        for frame_info in inspect.stack():
            frame = frame_info.frame
            module_name = frame.f_globals.get("__name__")
            if not module_name:
                continue

            # We only instrument the specific module that likely contains the classes
            # we want to augment (e.g., 'rag_based_llm'). Skip generic or stdlib modules.
            if module_name.endswith("rag_based_llm") or module_name.endswith("rag_based_llm.py"):
                module = sys.modules.get(module_name)
                if not module:
                    continue

                def _wrap_method(cls, method_name, pre_msg=None, post_msg=None, result_summary=None):
                    if not hasattr(cls, method_name):
                        return
                    # Avoid double-wrapping
                    flag = f"__logging_wrapped_{method_name}__"
                    if getattr(cls, flag, False):
                        return

                    orig = getattr(cls, method_name)

                    @functools.wraps(orig)
                    def wrapped(*args, **kwargs):
                        try:
                            if pre_msg:
                                try:
                                    if "%s" in pre_msg:
                                        logger.info(pre_msg, *args[1:])
                                    elif "{" in pre_msg:
                                        logger.info(pre_msg.format(*args[1:], **kwargs))
                                    else:
                                        logger.info(pre_msg)
                                except Exception:
                                    logger.info(pre_msg)
                            else:
                                logger.info("%s.%s called", cls.__name__, method_name)
                        except Exception:
                            # Best-effort logging only
                            pass
                        try:
                            result = orig(*args, **kwargs)
                        except Exception as e:
                            logger.exception("Exception in %s.%s: %s", cls.__name__, method_name, e)
                            raise
                        try:
                            if post_msg:
                                logger.info(post_msg.format(result=result))
                            elif result_summary and isinstance(result_summary, str):
                                try:
                                    # allow simple format like count of documents
                                    logger.info(result_summary.format(result=result))
                                except Exception:
                                    logger.debug("Result summary failed to format")
                        except Exception:
                            pass
                        return result

                    setattr(cls, method_name, wrapped)
                    setattr(cls, flag, True)

                # Candidate classes and methods to instrument
                for cls_name, methods in {
                    'RAGModel': [
                        ('build_config', "RAGModel.build_config starting for %s", "RAGModel.build_config completed", None),
                        ('ask', "RAGModel.ask called with question=%s, n_results=%s", "RAGModel.ask returned {result}", "Returned {result}")
                    ],
                    'Assistant': [
                        ('RAG_context_fetcher', "Assistant.RAG_context_fetcher called for question=%s", "Assistant.RAG_context_fetcher returned {result}", None),
                        ('chat_with_model', "Assistant.chat_with_model called with question=%s", "Assistant.chat_with_model returning question category: {result}", None)
                    ]
                }.items():
                    cls = getattr(module, cls_name, None)
                    if cls is None:
                        continue
                    for m in methods:
                        method_name = m[0]
                        pre_msg = m[1] if len(m) > 1 else None
                        post_msg = m[2] if len(m) > 2 else None
                        result_summary = m[3] if len(m) > 3 else None
                        try:
                            _wrap_method(cls, method_name, pre_msg=pre_msg, post_msg=post_msg, result_summary=result_summary)
                        except Exception:
                            pass
                break
    except Exception:
        # Keep instrumentation best-effort and non-fatal
        pass

    return logger
