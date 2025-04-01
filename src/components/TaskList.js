import React from "react";
import "../styles/TaskList.css";

function TaskList({ tasks, deleteTask, editTask, toggleComplete }) {
  return (
    <div className="task-list">
      <h2>Danh sách công việc</h2>
      <ul>
        {tasks.map((task) => (
          <li key={task.id} className={task.completed ? "completed" : ""}>
            {task.title} - {task.priority} ({task.category}) - 
            {task.dueDate ? `Hết hạn: ${new Date(task.dueDate).toLocaleDateString()}` : "Chưa có ngày hết hạn"}
            <button onClick={() => editTask(task)}>Chỉnh sửa</button>
            <button onClick={() => deleteTask(task.id)}>Xóa</button>
            <button onClick={() => toggleComplete(task.id)} className="check-btn">
              {task.completed ? "✓" : "✔"}
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default TaskList;
