import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory('sam2_segmentation')
    default_params = os.path.join(pkg_share, 'config', 'sam2_segmentation.yaml')

    params_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params,
        description='Path to the ROS 2 params YAML for the SAM 2 node.')

    node = Node(
        package='sam2_segmentation',
        executable='sam2_segmentation_node',
        name='sam2_segmentation_node',
        output='screen',
        parameters=[LaunchConfiguration('params_file')],
    )

    return LaunchDescription([params_arg, node])
