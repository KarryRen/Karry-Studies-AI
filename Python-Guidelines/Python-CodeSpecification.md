# Python-CodeSpecification

Python officially has a good style guide for Python code in [**PEP 8 - Style Guide for Python Code**](https://peps.python.org/pep-0008/) in the Python Enhancement Proposals (PEP). The Chinese translation of that document can be found [**HERE**](https://www.cnblogs.com/bymo/p/9567140.html) ! Please be sure to read this document carefully, as its contents are the CORE specification for python code development.

For code specification checking, we can use tools such as [**pylint**](https://zhuanlan.zhihu.com/p/364594654) and [**flake8**](https://github.com/pycqa/flake8). It is important to note that in python code development related to quantitative research we still focus on functionality and efficiency, so we **rarely use tools** to rigorously check code specifications. However, the basic norms of Python code development should not be ignored, and have the following main implications:

- **Facilitate communication**. Clear and standardized code brings focus to the core quickly.
- **Helps to review Pull Requests efficiently.** Focus the review on core logic changes to the code to minimize disruption due to code formatting.

Fortunately, many IDEs now have integrated PEP8-based code formatting tools within them:

- Allow us to format code with a **shortcut-key** 
  - For `PyCharm`, the [shortcut-key](https://blog.csdn.net/weixin_43250623/article/details/88829783) is **`Ctrl + Alt + L`  for Windows** or **`Option + Command + L` for Mac**
- Provide **real-time** code formatting **detection**
  - For `PyCharm`, it **displays PEP8-compliant comments** (usually a variety of wavy line hints) for non-standardized code, which you can follow on your own, or you can just use the shortcut keys mentioned above.

---

However, in our previous research work, we found that **some of the PEP8 requirements do not meet the actual needs**, so we propose the following targeted changes to the PEP8 specification:

**1. Adjust the maximum length of characters in the lines limit from [79](https://peps.python.org/pep-0008/#maximum-line-length) to 160.**

In algorithms such as factorization, where we generally write the very long factorization formulas, setting the maximum length of characters in the lines to 160 is a gread choice.

- For PyCharm, you can follow [**THIS**](https://blog.csdn.net/qq_38486203/article/details/126409118) to do the adjustment. `File → settings → editor → code style → Hard wrap at`

**2. Add file `.py` file header comments.**

Header comments will give the code a sense of unity while still providing very clear and relevant information. Please follow the template below for header comments.

```python
# -*- coding: utf-8 -*-
# @author : RenKai (intern in HIGGS ASSET)
# @time   : ${DATE} ${TIME}
#
# pylint: disable=no-member

""""""
```

- For PyCharm, you can follow [**THIS**](https://zhuanlan.zhihu.com/p/113896445) to add. `File → Settings →  Editor →  Code Style → File and Code Templates →  Python Script`

**3. About Strings.**

- Just use the `f""` to format all variables rather than `""%` or `"".format`
- Please use the `""` rather than `''`

