# Migrating from rclpy to synchros2

The `synchros2` programming model is a superset of `rclpy`'s. Code written for `rclpy`, which requires everything to be asynchronous, can be used with `synchros2` directly, while simultaneously enabling synchronous patterns. To do so, just:

* Use `@synchros2.process.main()` for your application entrypoint.
* Make sure to spin your node subclasses explicitly.

This allows you to migrate incrementally: you can keep your existing node subclasses, and still get access to a process-wide background executor for synchronous programming if desired.

We also recommend you subclass `synchros2.node.Node` instead of `rclpy.node.Node`. By default it is better behaved in multi-threaded settings.

## Example

**Before (on rclpy)**
```python
import rclpy
from rclpy.node import Node

class MyNode(Node):
	def __init__(self) -> None:
		super().__init__('my_node')
		# ... setup publishers, timers, etc ...

def main() -> None:
	rclpy.init()
	node = MyNode()
    try:
	    rclpy.spin(node)
	finally:
        node.destroy_node()
	    rclpy.try_shutdown()

if __name__ == '__main__':
	main()
```

**After (on synchros2)**
```python
import synchros2.process as ros_process
from synchros2.node import Node

class MyNode(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__('my_node', **kwargs)
        # ... setup publishers, timers, etc ...

@ros_process.main(prebaked=False)
def main() -> None:
    ros_process.spin(MyNode)

if __name__ == '__main__':
    main()
```

And if single-threaded execution is (still) a must to ensure equivalent behavior:

```python
from rclpy.executors import SingleThreadedExecutor

import synchros2.process as ros_process
from synchros2.executors import foreground
from synchros2.node import Node

class MyNode(Node):
    def __init__(self, **kwargs) -> None:
        super().__init__('my_node', **kwargs)
		# ... setup publishers, timers, etc ...

@ros_process.main(prebaked=False)
def main() -> None:
    with foreground(SingleThreadedExecutor()) as main.executor:
        main.spin(MyNode)

if __name__ == '__main__':
    main()
```
