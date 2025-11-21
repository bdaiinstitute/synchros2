# Executors and Callback Groups

`synchros2` uses and extends the same abstractions and patterns that `rclpy` offers to manage execution. Namely, _executors_ for task and callback dispatch and _callback groups_ for concurrency control. However, executors and callback groups provided by `rclpy` do not generally afford blocking calls. `synchros2` adds implementations of its own to better support these use cases.

## Executors

Executors in ROS 2 are components that manage how nodes' callbacks and tasks are scheduled and run. They handle the coordination of message processing, timers, and service requests by dispatching these events to one or more threads. Executors allow developers to control concurrency and execution flow.

### `rclpy.executors.SingleThreadedExecutor`

The simplest executor. Tasks and callbacks are dispatched and run by the same thread that is spinning it. 

```python
import rclpy

from rclpy.executors import SingleThreadedExecutor

def main():
    rclpy.init()
    node = rclpy.create_node("my_node")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()  # blocking!
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        executor.remove_node(node)
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == "__main__":
    main()
```

Concurrency is therefore impossible. A blocking call within a callback will block the entire executor, and if it relies on other callbacks to unblock it will invariably deadlock it. 

```python
import rclpy
from std_srvs.srv import Trigger

from rclpy.executors import SingleThreadedExecutor

def poll(client):
    print("About to deadlock...")
    client.call(Trigger.Request())  # deadlock!

def trigger_callback(request, response):
    response.success = True
    return response

def main():
    rclpy.init()
    node = rclpy.create_node("my_node")
    executor = SingleThreadedExecutor()
    node.create_service(Trigger, "trigger", trigger_callback)
    client = node.create_client(Trigger, "trigger")
    node.create_timer(1.0, lambda: poll(client))
    executor.add_node(node)
    try:
        executor.spin()  # blocking!
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        executor.remove_node(node)
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == "__main__":
    main()
```

On the flip side, lack of context switch makes the single threaded executor a good choice when time keeping is important (e.g. a node with a timer that needs to fire at a high rate).

```python
import time
import rclpy

from rclpy.executors import SingleThreadedExecutor

def main():
    rclpy.init()
    node = rclpy.create_node("my_node")
    executor = SingleThreadedExecutor()
    node.create_timer(0.1, lambda: print(time.time()))
    executor.add_node(node)
    try:
        executor.spin()  # blocking!
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        executor.remove_node(node)
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == "__main__":
    main()
```

### `rclpy.executors.MultiThreadedExecutor`

Simple threaded executor for concurrent execution. Tasks and callbacks are dispatched by the thread that is spinning it but run on a fixed size thread pool. 

```python
import time
import rclpy

from rclpy.executors import MultiThreadedExecutor

def timer_callback():
    print(time.time())
    time.sleep(0.5)

def main():
    rclpy.init()
    node = rclpy.create_node("my_node")
    node.create_timer(1.0, timer_callback)
    node.create_timer(1.0, timer_callback)
    executor = MultiThreadedExecutor(2)
    executor.add_node(node)
    try:
        executor.spin()  # blocking!
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        executor.remove_node(node)
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == "__main__":
    main()
```

Blocking calls are possible but its thread pool size must be chosen wisely. If there are fewer threads than standing blocking calls at any point in time, dispatch delays and deadlocks are back on the table.

### `synchros2.executors.AutoScalingMultiThreadedExecutor`

Akin to the `rclpy.executors.MultiThreadedExecutor` but managing thread pools that scale up and down on demand. Tasks and callbacks are dispatched by the thread that is spinning it but run on a thread pool that adds workers to match the demand. 

```python
import time
import rclpy

from synchros2.executors import AutoScalingMultiThreadedExecutor

def callback(i):
    print(f"[{time.time()}] Ahoy! (#{i})")

def main():
    rclpy.init()
    executor = AutoScalingMultiThreadedExecutor()
    for i in range(20):
        executor.create_task(callback, i)
    try:
        executor.spin()  # blocking!
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        rclpy.try_shutdown()

if __name__ == "__main__":
    main()
```

If the number of workers hits a configurable upper bound (`max_threads`), work is queued. If a worker is idle for too long (`max_thread_idle_time`), and the number of workers has not yet hit a configurable lower bound (0), it will be stopped. A concurrency quota can be configured for callback groups to avoid any given callback group from starving the rest (`max_threads_per_callback_group`).

By default, this executor manages a single thread pool, but more may be added. Callback groups can then be assigned to specific thread pools to further control callback dispatch and execution. 


```python
import time
import threading

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from synchros2.executors import AutoScalingMultiThreadedExecutor

def callback():
    current_thread = threading.current_thread()
    print(f"Ahoy! from {current_thread.name}")

def timer_callback(executor):
    current_thread = threading.current_thread()
    print(f"Dispatch from {current_thread.name} at time {time.time()}")
    for i in range(10):
        executor.create_task(callback)

def main():
    rclpy.init()
    node = rclpy.create_node("my_node")
    executor = AutoScalingMultiThreadedExecutor()
    thread_pool = executor.add_static_thread_pool(1)
    custom_callback_group = MutuallyExclusiveCallbackGroup()
    executor.bind(custom_callback_group, thread_pool)
    node.create_timer(0.1, lambda: timer_callback(executor), custom_callback_group)
    executor.add_node(node)
    try:
        executor.spin()  # blocking!
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        executor.remove_node(node)
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == "__main__":
    main()
```

## Callback groups

Callback groups in ROS 2 are mechanisms that control how callbacks (functions triggered by events like messages or timers) are scheduled and executed within a node. They let you specify concurrency rules, such as whether callbacks can run simultaneously or must run one at a time. There is little value to them in single threaded execution scenarios but they are crucial when using multi-threaded executors. By assigning callbacks to different groups, you can fine-tune how your system handles incoming events and tasks.

### `rclpy.callback_groups.MutuallyExclusiveCallbackGroup`

Tasks and callbacks may only run one at a time when in this callback group. 

This is the default callback group for `rclpy.node.Node` instances.

### `rclpy.callback_groups.ReentrantCallbackGroup`

Tasks and callbacks may run concurrently when in this callback group, including multiple instances of the same callback e.g. when a topic subscription message arrives before it is done processing the previous one.

### `synchros2.callback_groups.NonReentrantCallbackGroup`

Tasks and callbacks may run concurrently when in this callback group, but there may only be one instance of a given callback running at any given time e.g. topic subscription messages will always be processed sequentially.

This is the default callback group for `synchros2.node.Node` instances.

## Utilities

To simplify executor management, `synchros2` offers a few additional utilities.

If you need to spin in foreground (i.e. block on the executor), you may:
```python
from rclpy.executors import SingleThreadedExecutor
from synchros2.executors import foreground

with foreground(SingleThreadedExecutor()) as executor:
    executor.spin()
```

If you need to spin in the background (i.e. keep the executor running while blocking on something else), you may:
```python
from rclpy.executors import SingleThreadedExecutor
from synchros2.executors import background

with background(SingleThreadedExecutor()) as executor:
    # do work, no need to spin
    # executor.spin() would raise 
```

Either way, `synchros2` context managers will take care of the executor lifecycle for you.
