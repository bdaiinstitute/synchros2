# Repository Migration Notice

> [!IMPORTANT]
> This repository is moving to the **RAI-Opensource** GitHub organization in **about 2 weeks from April 9, 2026 (UTC)**.
>
> Current status: **pre-migration**.
> Migration target org: <https://github.com/RAI-Opensource>
>
> What to do now:
>
> - Watch this repository for the final cutover update and destination repository link.
> - Plan to update your git remote after the move:
>
>   ```bash
>   git remote set-url origin <new-repository-url>
>   ```
>
> - If you maintain downstream docs/scripts, prepare to update links from this repository to the new org.
>

# `synchros2`

![Python Support](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)
![ROS Support](https://img.shields.io/badge/ROS-humble%20%7C%20jazzy-blue)

## Overview

`synchros2` enables a different, at times simpler approach to ROS 2 programming, particularly for those that come with a ROS 1 background.  See the [`synchros2` README](synchros2/README.md) for more information.

## Packages

This repository contains the following packages:

| Package                                                                             | Description                                                                   |
|-------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| [`synchros2`](synchros2)                                                            | `rclpy` wrappers to ease idiomatic, synchronous ROS 2 programming in Python.  |
| [`synchros2_tutorials_example`](synchros2_tutorials_example)                        | Support code for `synchros2` tutorials.                                       |
| [`synchros2_tutorials_interfaces_example`](synchros2_tutorials_interfaces_example)  | Support interfaces for `synchros2` tutorials.                                 |

## Next steps

See [contribution guidelines](CONTRIBUTING.md)!
