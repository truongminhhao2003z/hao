import React, { useState, useEffect } from "react";
import "../styles/TaskForm.css";

function TaskForm({ addTask, editTask, taskToEdit }) {
  const [task, setTask] = useState({
    title: "",
    priority: "Cao",
    category: "Công việc",
    dueDate: "",
    completed: false,
  });

  useEffect(() => {
    if (taskToEdit) {
      setTask(taskToEdit);
    }
  }, [taskToEdit]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setTask({ ...task, [name]: value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (taskToEdit) {
      editTask(task);
    } else {
      addTask({ ...task, id: Date.now() });
    }
    setTask({ title: "", priority: "Cao", category: "Công việc", dueDate: "", completed: false });
  };

  return (
    <form onSubmit={handleSubmit} className="task-form">
      <input
        type="text"
        name="title"
        placeholder="Tên công việc new"
        value={task.title}
        onChange={handleChange}
        required
      />
      <select name="priority" value={task.priority} onChange={handleChange}>
        <option value="Cao">Cao</option>
        <option value="Trung bình">Trung bình</option>
        <option value="Thấp">Thấp</option>
      </select>
      <select name="category" value={task.category} onChange={handleChange}>
        <option value="Công việc">Công việc</option>
        <option value="Học tập">Học tập</option>
        <option value="Cá nhân">Cá nhân</option>
      </select>
      {/* Ngày hết hạn */}
      <input
        type="date"
        name="dueDate"
        value={task.dueDate}
        onChange={handleChange}
      />
      <button type="submit">{taskToEdit ? "Chỉnh sửa công việc" : "Thêm công việc"}</button>
    </form>
  );
}

export default TaskForm;
