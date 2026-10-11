import { useEffect, useState } from "react";
import { api } from "../services/api";

export function useHealth() {
  const [status, setStatus] = useState("offline");
  useEffect(() => {
    api.health().then((h) => setStatus(h.status)).catch(() => setStatus("offline"));
  }, []);
  return status;
}
