*************
API Reference
*************
This page provides direct access to the detailed API documentation for each module in the xTrack system.

Key Classes
===========

PersonTracker
-------------

The main tracking class that orchestrates the entire system.

.. autoclass:: person_tracker.PersonTracker
   :members:
   :undoc-members:
   :show-inheritance:

VestClassifier
--------------

Model-based vest detection using PyTorch.

.. autoclass:: vest_classifier.VestClassifier
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions
=============

Person Detection and Tracking
-----------------------------

.. autofunction:: person_tracker.PersonTracker._process_frame
.. autofunction:: person_tracker.PersonTracker._handle_custom_reid
.. autofunction:: person_tracker.PersonTracker._process_detection

Vest Detection
--------------

.. autofunction:: vision.detect_yellow_vest
.. autofunction:: vision.extract_upper_body_roi
.. autofunction:: vision.validate_hsv_color

Re-Identification
-----------------

.. autofunction:: reid.get_appearance_embedding
.. autofunction:: reid.cosine_similarity
.. autofunction:: reid.initialize_reid_model

3D Localization
---------------

.. autofunction:: localization.get_3d_position
.. autofunction:: localization.get_closest_depth_frame
.. autofunction:: lidar_utils.get_3d_position_from_lidar
.. autofunction:: lidar_utils.fuse_depth_and_lidar

LiDAR Processing
----------------

.. autofunction:: lidar_utils.load_point_cloud
.. autofunction:: lidar_utils.filter_point_cloud
.. autofunction:: lidar_utils.remove_ground_plane
.. autofunction:: lidar_utils.cluster_points_in_box

Configuration and Utilities
---------------------------

.. autofunction:: config.get_dataset_paths
.. autofunction:: utils.parse_arguments
.. autofunction:: utils.load_timestamps
.. autofunction:: utils.log_result


