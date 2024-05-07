# Submodule-Guidelines

在开发比较复杂的项目时，我们有可能会将代码根据功能拆解成不同的子模块。主项目对子模块有依赖，却又并不关心子模块的内部开发流程细节，同时子模块开发相对独立且秘密。这种情况下，通常不会把所有源码都放在同一个 Git 仓库中，而是选择使用 Submodule 这一 Git 工具进行管理。此处的相关说明参考 [**Ref.**](https://zhuanlan.zhihu.com/p/87053283) 将研究开发中常用的内容拿出来，如果您有其他具体的问题可以仔细阅读该参考。

## How to use submodule ?

为了让 Submodule 的使用过程更加清晰，我们在此以一个具体的例子做阐述。假定有两个仓库：

- 主项目：`repo-main`，远程仓库地址为 `https://github.com/username/repo-main.git`
- 子模块：`repo-sub-1`，远程仓库地址为 `https://github.com/username/repo-sub-1.git`

希望在 `repo-main` 中添加 `repo-sub-1` ，而又保持 `repo-sub-1` 自身独立的版本控制。

### How to add the submodule ?

将子模块 `repo-sub-1` 添加到主项目 `repo-main` 的过程较为简单，只需要进入到主项目 `repo-main` 中然后进行添加即可：

```shell
cd repo-main
git submodule add <submodule_url: https://github.com/username/repo-sub-1.git>
```

此时主项目仓库  `repo-main` 中会多出两个文件：`.gitmodules` 和 `repo-sub-1` ，其中：

- `.gitmodules` 就是子模块的相关控制信息
- `repo-sub-1` 就是子模块远程仓库中的所有文件

### How to clone the repo with submodule ?

实际中您可能更多遇到的情况是：对一个已经添加好 Submodule 的仓库进行继续开发。例如对于已经添加好子模块 `repo-sub-1` 的主项目 `repo-main` 进行开发，此时主项目的远程仓库结构如下：

```python
repo-main
├── repo-sub-1@xxx
├── .gitmodules
└── other_files
```

可以看到该仓库中已经有了 `.gitmodules` 和 `repo-sub-1@xxx` 两个文件，这时您如果点击 `repo-sub-1@xxx`，就会自动跳转到子模块的远程仓库下。如果你想将该仓库 `Clone` 到本地，请按照如下步骤：

```shell
# ---- Step 1. Clone the repo without submodules ---- #
git clone <main_url: https://github.com/username/repo-main.git> 
# 完成第一步后你会发现, 虽然本地的 `repo-main/` 文件夹下已经有了 `.gitmodules` 和 `repo-sub-1/`
# 但是 `repo-sub-1/` 文件夹下没有任何的文件, 和 `repo-sub-1` 的远程仓库不一致

# ---- Step 2. Init and update the submodules ---- #
git submodule init # init the submodule
git submodule update repo-sub-1 # update the submodule
# 完成初始化和子模块更新后, 你就发现 `repo-sub-1/` 文件和自动跳转到的子模块远程仓库下的文件完全一致了
```

### How to update the submodule in main repo ?

`repo-main` 和 `repo-sub-1` 总是在更新，注意 :warning: ：

- 所有想对子模块 `repo-sub-1` 内容的更新，请在 `repo-sub-1` 中进行 add , commit 和 push 永远不要在 `repo-main/repo-sub-1` 下更新并尝试 push，这样做是没有意义的 ！
- 所有想要对主项目 `repo-main` 内容的更新，请直接在 `repo-main` 中进行 add , commit 和 push，就像你平时做得那样。

您肯定会发现，就算对 `repo-sub-1` 子模块的内容完成了 push ，主模块 `repo-main` 仓库仍然会链接（点击 `repo-sub-1@xxx` 到达的仓库地址）到之前没有变动的版本，这意味着您主模块 `repo-main` 本地的内容也不会发生任何改变。如果您想更新主模块 `repo-main` 下的子模块 `repo-sub-1`，请按照如下步骤：

```shell
# ---- Step 0. Check your work directory ---- #
# 请先仔细检查当前命令行的工作路径, 确保当前在 `repo-main/` 根目录下

# ---- Step 1. Update the submodule ---- #
git submodule update repo-main/repo-sub-1

# ---- Step 2. Pull the latest submodule file ---- #
cd repo-sub-1
git pull origin master
# .... Finish pull

# ---- Step 3. Go back to the root directory and update the link of submodule ---- #
cd .. 
git add .
git commit -m "change repo-sub-1 version"
git push
```



## Some attentions about submodule.

- 永远不要在主项目下更新子模块内容并尝试 push，这样的操作是没有意义的
- 如果子模块内容完成了更新，请及时修改主项目对子项目的链接



## Some bugs we have suffered.

> 这里是我们之前在使用 `Submodule` 时出现的一些 Bug 和解决方案

**1. 子模块在分支上更新并完成 Merge 时，在主项目中完成对子模块更新后，出现子模块 Page Not Found 的情况**

- **出现的原因**：主项目中对子模块最新 commit id 的链接出错，主项目 @commit id 和子模块最新的 commit id 不一致。

- **解决的方案**：在主项目中进行如下操作

  ```shell
  # ---- Step 1. Enter the submodule and reset the commit id ---- #
  cd repo-sub-1
  git log # use log to check the right commit id of submodule
  git reset `right_commit_id`
  
  # ---- Step 2. Go back to the root directory and update the link of submodule ---- #
  cd .. 
  git add .
  git commit -m "change repo-sub-1 version"
  git push
  ```

