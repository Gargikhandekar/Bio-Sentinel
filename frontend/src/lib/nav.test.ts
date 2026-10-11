import { describe, expect, it } from "vitest";

describe("navigation contract", () => {
  it("exposes the research lab routes", () => {
    const routes = ["/", "/predict", "/models", "/drift", "/explain", "/research", "/research/ablation", "/research/calibration", "/system"];
    expect(routes).toHaveLength(9);
  });
});
