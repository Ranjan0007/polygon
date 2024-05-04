import cv2
import numpy as np

class PolygonEditor:
    def __init__(self, image):
        self.original_image = image.copy()
        self.image = image.copy()
        self.polygons = []
        self.selected_polygon_index = None
        self.dragging_vertex_index = None

    def start(self):
        cv2.namedWindow("Polygon Editor")
        cv2.setMouseCallback("Polygon Editor", self.mouse_callback)

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord("e"):
                self.detect_edges()

            elif key == ord("p"):
                self.convert_edges_to_polygons()

            elif key == ord("j"):
                self.select_previous_polygon()

            elif key == ord("k"):
                self.select_next_polygon()

            elif key == ord("a") or key == ord("s") or key == ord("w") or key == ord("d"):
                self.move_vertex(key)

            elif key == ord("r"):
                self.create_approximate_rectangle()

            elif key == 27:  # Esc key to exit
                break

            cv2.imshow("Polygon Editor", self.image)

        cv2.destroyAllWindows()

    def detect_edges(self):
        edges = cv2.Canny(cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY), 50, 150)
        self.image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    def convert_edges_to_polygons(self):
        contours, _ = cv2.findContours(cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.polygons = [cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True) for contour in contours]
        self.draw_polygons()

    def draw_polygons(self):
        self.image = self.original_image.copy()
        for i, polygon in enumerate(self.polygons):
            if i == self.selected_polygon_index:
                cv2.drawContours(self.image, [polygon], 0, (0, 255, 0), 2)
            else:
                cv2.drawContours(self.image, [polygon], 0, (255, 0, 0), 2)

        if self.selected_polygon_index is not None:
            for i, point in enumerate(self.polygons[self.selected_polygon_index]):
                if i == self.dragging_vertex_index:
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 255)
                cv2.circle(self.image, (point[0][0], point[0][1]), 5, color, -1)

    def select_previous_polygon(self):
        if self.selected_polygon_index is None:
            self.selected_polygon_index = 0
        else:
            self.selected_polygon_index = (self.selected_polygon_index - 1) % len(self.polygons)
        self.draw_polygons()

    def select_next_polygon(self):
        if self.selected_polygon_index is None:
            self.selected_polygon_index = 0
        else:
            self.selected_polygon_index = (self.selected_polygon_index + 1) % len(self.polygons)
        self.draw_polygons()

    def move_vertex(self, key):
        if self.selected_polygon_index is not None and self.dragging_vertex_index is not None:
            point = self.polygons[self.selected_polygon_index][self.dragging_vertex_index][0]
            if key == ord("a"):
                point[0] -= 1
            elif key == ord("d"):
                point[0] += 1
            elif key == ord("w"):
                point[1] -= 1
            elif key == ord("s"):
                point[1] += 1
            self.draw_polygons()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.selected_polygon_index is not None:
                for i, point in enumerate(self.polygons[self.selected_polygon_index]):
                    distance = np.sqrt((point[0][0] - x) ** 2 + (point[0][1] - y) ** 2)
                    if distance < 5:
                        self.dragging_vertex_index = i
                        break
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging_vertex_index = None
        elif event == cv2.EVENT_MOUSEMOVE:
            self.move_vertex(cv2.waitKey(1))

    def create_approximate_rectangle(self):
        if self.selected_polygon_index is not None:
            polygon = self.polygons[self.selected_polygon_index]
            rect = cv2.minAreaRect(polygon)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(self.image, [box], 0, (0, 255, 255), 2)

            area_polygon = cv2.contourArea(polygon)
            area_rect = cv2.contourArea(box)
            intersection = cv2.contourArea(cv2.convexHull(polygon))

            print("Area of Polygon:", area_polygon)
            print("Area of Rectangle:", area_rect)
            print("Area of Intersection:", intersection)

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        editor = PolygonEditor(frame)
        editor.start()

    cap.release()
    cv2.destroyAllWindows()

