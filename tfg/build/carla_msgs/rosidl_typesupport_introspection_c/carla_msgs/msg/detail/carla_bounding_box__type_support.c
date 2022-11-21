// generated from rosidl_typesupport_introspection_c/resource/idl__type_support.c.em
// with input from carla_msgs:msg/CarlaBoundingBox.idl
// generated code does not contain a copyright notice

#include <stddef.h>
#include "carla_msgs/msg/detail/carla_bounding_box__rosidl_typesupport_introspection_c.h"
#include "carla_msgs/msg/rosidl_typesupport_introspection_c__visibility_control.h"
#include "rosidl_typesupport_introspection_c/field_types.h"
#include "rosidl_typesupport_introspection_c/identifier.h"
#include "rosidl_typesupport_introspection_c/message_introspection.h"
#include "carla_msgs/msg/detail/carla_bounding_box__functions.h"
#include "carla_msgs/msg/detail/carla_bounding_box__struct.h"


// Include directives for member types
// Member `center`
// Member `size`
#include "geometry_msgs/msg/vector3.h"
// Member `center`
// Member `size`
#include "geometry_msgs/msg/detail/vector3__rosidl_typesupport_introspection_c.h"

#ifdef __cplusplus
extern "C"
{
#endif

void CarlaBoundingBox__rosidl_typesupport_introspection_c__CarlaBoundingBox_init_function(
  void * message_memory, enum rosidl_runtime_c__message_initialization _init)
{
  // TODO(karsten1987): initializers are not yet implemented for typesupport c
  // see https://github.com/ros2/ros2/issues/397
  (void) _init;
  carla_msgs__msg__CarlaBoundingBox__init(message_memory);
}

void CarlaBoundingBox__rosidl_typesupport_introspection_c__CarlaBoundingBox_fini_function(void * message_memory)
{
  carla_msgs__msg__CarlaBoundingBox__fini(message_memory);
}

static rosidl_typesupport_introspection_c__MessageMember CarlaBoundingBox__rosidl_typesupport_introspection_c__CarlaBoundingBox_message_member_array[2] = {
  {
    "center",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(carla_msgs__msg__CarlaBoundingBox, center),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  },
  {
    "size",  // name
    rosidl_typesupport_introspection_c__ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    NULL,  // members of sub message (initialized later)
    false,  // is array
    0,  // array size
    false,  // is upper bound
    offsetof(carla_msgs__msg__CarlaBoundingBox, size),  // bytes offset in struct
    NULL,  // default value
    NULL,  // size() function pointer
    NULL,  // get_const(index) function pointer
    NULL,  // get(index) function pointer
    NULL  // resize(index) function pointer
  }
};

static const rosidl_typesupport_introspection_c__MessageMembers CarlaBoundingBox__rosidl_typesupport_introspection_c__CarlaBoundingBox_message_members = {
  "carla_msgs__msg",  // message namespace
  "CarlaBoundingBox",  // message name
  2,  // number of fields
  sizeof(carla_msgs__msg__CarlaBoundingBox),
  CarlaBoundingBox__rosidl_typesupport_introspection_c__CarlaBoundingBox_message_member_array,  // message members
  CarlaBoundingBox__rosidl_typesupport_introspection_c__CarlaBoundingBox_init_function,  // function to initialize message memory (memory has to be allocated)
  CarlaBoundingBox__rosidl_typesupport_introspection_c__CarlaBoundingBox_fini_function  // function to terminate message instance (will not free memory)
};

// this is not const since it must be initialized on first access
// since C does not allow non-integral compile-time constants
static rosidl_message_type_support_t CarlaBoundingBox__rosidl_typesupport_introspection_c__CarlaBoundingBox_message_type_support_handle = {
  0,
  &CarlaBoundingBox__rosidl_typesupport_introspection_c__CarlaBoundingBox_message_members,
  get_message_typesupport_handle_function,
};

ROSIDL_TYPESUPPORT_INTROSPECTION_C_EXPORT_carla_msgs
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, carla_msgs, msg, CarlaBoundingBox)() {
  CarlaBoundingBox__rosidl_typesupport_introspection_c__CarlaBoundingBox_message_member_array[0].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Vector3)();
  CarlaBoundingBox__rosidl_typesupport_introspection_c__CarlaBoundingBox_message_member_array[1].members_ =
    ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_c, geometry_msgs, msg, Vector3)();
  if (!CarlaBoundingBox__rosidl_typesupport_introspection_c__CarlaBoundingBox_message_type_support_handle.typesupport_identifier) {
    CarlaBoundingBox__rosidl_typesupport_introspection_c__CarlaBoundingBox_message_type_support_handle.typesupport_identifier =
      rosidl_typesupport_introspection_c__identifier;
  }
  return &CarlaBoundingBox__rosidl_typesupport_introspection_c__CarlaBoundingBox_message_type_support_handle;
}
#ifdef __cplusplus
}
#endif
