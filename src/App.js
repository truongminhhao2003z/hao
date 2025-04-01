import React, { useState, useEffect } from "react";
import TaskForm from "./components/TaskForm";
import "./App.css";
import { DragDropContext, Droppable, Draggable } from "react-beautiful-dnd";

function App() {
  // Đọc dữ liệu từ localStorage khi ứng dụng khởi động lại
  const [tasks, setTasks] = useState(() => {
    const savedTasks = localStorage.getItem("tasks");
    return savedTasks ? JSON.parse(savedTasks) : [];
  });

  const [taskToEdit, setTaskToEdit] = useState(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [filterPriority, setFilterPriority] = useState("all");
  const [filterCategory, setFilterCategory] = useState("all");
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [notes, setNotes] = useState({}); // Lưu ghi chú của mỗi công việc
  const [editingNoteId, setEditingNoteId] = useState(null); // ID của công việc đang chỉnh sửa ghi chú
  const [newNote, setNewNote] = useState(""); // Lưu ghi chú mới khi chỉnh sửa

  // Lưu trạng thái chế độ Dark Mode vào localStorage
  useEffect(() => {
    const savedMode = localStorage.getItem("darkMode");
    if (savedMode) {
      setIsDarkMode(JSON.parse(savedMode));
    }
  }, []);

  // Cập nhật chế độ Dark Mode trong localStorage
  useEffect(() => {
    localStorage.setItem("darkMode", JSON.stringify(isDarkMode));
    if (isDarkMode) {
      document.body.classList.add("dark-mode");
    } else {
      document.body.classList.remove("dark-mode");
    }
  }, [isDarkMode]);

  // Lưu công việc vào localStorage mỗi khi thay đổi
  useEffect(() => {
    localStorage.setItem("tasks", JSON.stringify(tasks));
  }, [tasks]);

  const addTask = (newTask) => {
    const updatedTasks = [...tasks, newTask];
    setTasks(updatedTasks); // Sẽ lưu vào localStorage nhờ useEffect
  };

  const deleteTask = (taskId) => {
    const updatedTasks = tasks.filter((task) => task.id !== taskId);
    setTasks(updatedTasks); // Sẽ lưu vào localStorage nhờ useEffect
  };

  const editTask = (updatedTask) => {
    const updatedTasks = tasks.map((task) =>
      task.id === updatedTask.id ? updatedTask : task
    );
    setTasks(updatedTasks); // Sẽ lưu vào localStorage nhờ useEffect
    setTaskToEdit(null); // Reset form sau khi chỉnh sửa
  };

  const toggleComplete = (taskId) => {
    const updatedTasks = tasks.map((task) =>
      task.id === taskId ? { ...task, completed: !task.completed } : task
    );
    setTasks(updatedTasks); // Sẽ lưu vào localStorage nhờ useEffect
  };

  const handleNoteClick = (taskId) => {
    setEditingNoteId(taskId); // Chuyển sang chế độ chỉnh sửa ghi chú
    setNewNote(notes[taskId] || ""); // Hiển thị ghi chú hiện tại nếu có
  };

  const handleNoteSave = (taskId) => {
    const updatedNotes = {
      ...notes,
      [taskId]: newNote, // Lưu ghi chú mới vào trạng thái notes
    };
    setNotes(updatedNotes); // Cập nhật lại ghi chú
    setEditingNoteId(null); // Đóng chế độ chỉnh sửa
    setNewNote(""); // Reset nội dung ghi chú
  };

  const handleNoteChange = (e) => {
    setNewNote(e.target.value); // Cập nhật giá trị ghi chú khi thay đổi
  };

  const filteredTasks = tasks
    .filter((task) => {
      return task.title.toLowerCase().includes(searchTerm.toLowerCase());
    })
    .filter((task) => {
      if (filterPriority === "all") return true;
      return task.priority === filterPriority;
    })
    .filter((task) => {
      if (filterCategory === "all") return true;
      return task.category === filterCategory;
    });

  // Hàm xử lý sự kiện kéo thả
  const onDragEnd = (result) => {
    const { destination, source } = result;
    if (!destination) return;

    const items = Array.from(tasks);
    const [reorderedItem] = items.splice(source.index, 1);
    items.splice(destination.index, 0, reorderedItem);

    setTasks(items); // Lưu lại thứ tự mới trong localStorage
  };

  return (
    <div className="container">
      <h1>Quản lý công việc</h1>
      {/* Nút chuyển đổi Dark Mode */}
      <button
        className="dark-mode-toggle"
        onClick={() => setIsDarkMode(!isDarkMode)}
      >
        <i className={isDarkMode ? "fas fa-sun" : "fas fa-moon"}></i>
      </button>

      {/* Thanh tìm kiếm */}
      <input
        type="text"
        placeholder="Tìm kiếm công việc..."
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
      />

      {/* Lọc theo mức độ ưu tiên */}
      <select onChange={(e) => setFilterPriority(e.target.value)} value={filterPriority}>
        <option value="all">Tất cả mức độ</option>
        <option value="Cao">Cao</option>
        <option value="Trung bình">Trung bình</option>
        <option value="Thấp">Thấp</option>
      </select>

      {/* Lọc theo danh mục */}
      <select onChange={(e) => setFilterCategory(e.target.value)} value={filterCategory}>
        <option value="all">Tất cả danh mục</option>
        <option value="Công việc">Công việc</option>
        <option value="Học tập">Học tập</option>
        <option value="Cá nhân">Cá nhân</option>
      </select>

      <TaskForm addTask={addTask} editTask={editTask} taskToEdit={taskToEdit} />

      {/* Danh sách công việc với kéo thả */}
      <DragDropContext onDragEnd={onDragEnd}>
        <Droppable droppableId="tasks">
          {(provided) => (
            <ul
              className="task-list"
              {...provided.droppableProps}
              ref={provided.innerRef}
            >
              {filteredTasks.map((task, index) => (
                <Draggable key={task.id} draggableId={task.id.toString()} index={index}>
                  {(provided) => (
                    <li
                      className={`task-item ${task.completed ? "completed" : ""}`}
                      ref={provided.innerRef}
                      {...provided.draggableProps}
                      {...provided.dragHandleProps}
                    >
                      <div className="task-content">
                        {/* Nội dung công việc */}
                        <div>{task.title}</div>
                      </div>
                      <div className="task-actions">
                        <button onClick={() => toggleComplete(task.id)}>
                          Hoàn thành
                        </button>
                        <button onClick={() => deleteTask(task.id)}>Xóa</button>
                        <button onClick={() => setTaskToEdit(task)}>Sửa</button>
                        {/* Nút ghi chú */}
                        <button
                          className="note-button"
                          onClick={() => handleNoteClick(task.id)}
                        >
                          Ghi chú
                        </button>
                      </div>
                      {/* Hiển thị ghi chú hoặc chế độ chỉnh sửa ghi chú */}
                      {editingNoteId === task.id ? (
                        <div className="note-edit">
                          <textarea
                            value={newNote}
                            onChange={handleNoteChange}
                          />
                          <button onClick={() => handleNoteSave(task.id)}>Lưu</button>
                        </div>
                      ) : (
                        notes[task.id] && (
                          <div className="note-display">
                            <p>{notes[task.id]}</p>
                          </div>
                        )
                      )}
                    </li>
                  )}
                </Draggable>
              ))}
              {provided.placeholder}
            </ul>
          )}
        </Droppable>
      </DragDropContext>
    </div>
  );
}

export default App;
