// generated from rosidl_typesupport_fastrtps_cpp/resource/idl__type_support.cpp.em
// with input from carla_ros_scenario_runner_types:msg/CarlaScenarioList.idl
// generated code does not contain a copyright notice
#include "carla_ros_scenario_runner_types/msg/detail/carla_scenario_list__rosidl_typesupport_fastrtps_cpp.hpp"
#include "carla_ros_scenario_runner_types/msg/detail/carla_scenario_list__struct.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_fastrtps_cpp/identifier.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_fastrtps_cpp/wstring_conversion.hpp"
#include "fastcdr/Cdr.h"


// forward declaration of message dependencies and their conversion functions
namespace carla_ros_scenario_runner_types
{
namespace msg
{
namespace typesupport_fastrtps_cpp
{
bool cdr_serialize(
  const carla_ros_scenario_runner_types::msg::CarlaScenario &,
  eprosima::fastcdr::Cdr &);
bool cdr_deserialize(
  eprosima::fastcdr::Cdr &,
  carla_ros_scenario_runner_types::msg::CarlaScenario &);
size_t get_serialized_size(
  const carla_ros_scenario_runner_types::msg::CarlaScenario &,
  size_t current_alignment);
size_t
max_serialized_size_CarlaScenario(
  bool & full_bounded,
  size_t current_alignment);
}  // namespace typesupport_fastrtps_cpp
}  // namespace msg
}  // namespace carla_ros_scenario_runner_types


namespace carla_ros_scenario_runner_types
{

namespace msg
{

namespace typesupport_fastrtps_cpp
{

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_carla_ros_scenario_runner_types
cdr_serialize(
  const carla_ros_scenario_runner_types::msg::CarlaScenarioList & ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Member: scenarios
  {
    size_t size = ros_message.scenarios.size();
    cdr << static_cast<uint32_t>(size);
    for (size_t i = 0; i < size; i++) {
      carla_ros_scenario_runner_types::msg::typesupport_fastrtps_cpp::cdr_serialize(
        ros_message.scenarios[i],
        cdr);
    }
  }
  return true;
}

bool
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_carla_ros_scenario_runner_types
cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  carla_ros_scenario_runner_types::msg::CarlaScenarioList & ros_message)
{
  // Member: scenarios
  {
    uint32_t cdrSize;
    cdr >> cdrSize;
    size_t size = static_cast<size_t>(cdrSize);
    ros_message.scenarios.resize(size);
    for (size_t i = 0; i < size; i++) {
      carla_ros_scenario_runner_types::msg::typesupport_fastrtps_cpp::cdr_deserialize(
        cdr, ros_message.scenarios[i]);
    }
  }

  return true;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_carla_ros_scenario_runner_types
get_serialized_size(
  const carla_ros_scenario_runner_types::msg::CarlaScenarioList & ros_message,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Member: scenarios
  {
    size_t array_size = ros_message.scenarios.size();

    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);

    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        carla_ros_scenario_runner_types::msg::typesupport_fastrtps_cpp::get_serialized_size(
        ros_message.scenarios[index], current_alignment);
    }
  }

  return current_alignment - initial_alignment;
}

size_t
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_PUBLIC_carla_ros_scenario_runner_types
max_serialized_size_CarlaScenarioList(
  bool & full_bounded,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;
  (void)full_bounded;


  // Member: scenarios
  {
    size_t array_size = 0;
    full_bounded = false;
    current_alignment += padding +
      eprosima::fastcdr::Cdr::alignment(current_alignment, padding);


    for (size_t index = 0; index < array_size; ++index) {
      current_alignment +=
        carla_ros_scenario_runner_types::msg::typesupport_fastrtps_cpp::max_serialized_size_CarlaScenario(
        full_bounded, current_alignment);
    }
  }

  return current_alignment - initial_alignment;
}

static bool _CarlaScenarioList__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  auto typed_message =
    static_cast<const carla_ros_scenario_runner_types::msg::CarlaScenarioList *>(
    untyped_ros_message);
  return cdr_serialize(*typed_message, cdr);
}

static bool _CarlaScenarioList__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  auto typed_message =
    static_cast<carla_ros_scenario_runner_types::msg::CarlaScenarioList *>(
    untyped_ros_message);
  return cdr_deserialize(cdr, *typed_message);
}

static uint32_t _CarlaScenarioList__get_serialized_size(
  const void * untyped_ros_message)
{
  auto typed_message =
    static_cast<const carla_ros_scenario_runner_types::msg::CarlaScenarioList *>(
    untyped_ros_message);
  return static_cast<uint32_t>(get_serialized_size(*typed_message, 0));
}

static size_t _CarlaScenarioList__max_serialized_size(bool & full_bounded)
{
  return max_serialized_size_CarlaScenarioList(full_bounded, 0);
}

static message_type_support_callbacks_t _CarlaScenarioList__callbacks = {
  "carla_ros_scenario_runner_types::msg",
  "CarlaScenarioList",
  _CarlaScenarioList__cdr_serialize,
  _CarlaScenarioList__cdr_deserialize,
  _CarlaScenarioList__get_serialized_size,
  _CarlaScenarioList__max_serialized_size
};

static rosidl_message_type_support_t _CarlaScenarioList__handle = {
  rosidl_typesupport_fastrtps_cpp::typesupport_identifier,
  &_CarlaScenarioList__callbacks,
  get_message_typesupport_handle_function,
};

}  // namespace typesupport_fastrtps_cpp

}  // namespace msg

}  // namespace carla_ros_scenario_runner_types

namespace rosidl_typesupport_fastrtps_cpp
{

template<>
ROSIDL_TYPESUPPORT_FASTRTPS_CPP_EXPORT_carla_ros_scenario_runner_types
const rosidl_message_type_support_t *
get_message_type_support_handle<carla_ros_scenario_runner_types::msg::CarlaScenarioList>()
{
  return &carla_ros_scenario_runner_types::msg::typesupport_fastrtps_cpp::_CarlaScenarioList__handle;
}

}  // namespace rosidl_typesupport_fastrtps_cpp

#ifdef __cplusplus
extern "C"
{
#endif

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_cpp, carla_ros_scenario_runner_types, msg, CarlaScenarioList)() {
  return &carla_ros_scenario_runner_types::msg::typesupport_fastrtps_cpp::_CarlaScenarioList__handle;
}

#ifdef __cplusplus
}
#endif
