torch.setdefaulttensortype('torch.FloatTensor')

local function dot(p1, p2)
    return p1.x * p2.x + p1.y * p2.y
end

local function vector(p1, p2)
    return {x=p2.x-p1.x, y=p2.y-p1.y}
end

local function scaleVector(v, scale)
    return {x=v.x * scale, y=v.y * scale}
end

local function addVector(v1, v2)
    return {x=v1.x+v2.x, y=v1.y+v2.y}
end

local function len(p1, p2)
    local v = vector(p1, p2)
    return math.sqrt(dot(v,v))
end

local function lineDist(line_pt1, line_pt2, Point)
    local line = vector(line_pt1, line_pt2)
    local l2 = dot(line,line)
    if l2 == 0.0 then return len(line_pt1, Point) end
    -- consider the line extending the segment, parameterized as p1 + t(p2-p1)
    -- find projection of point p onto the line, it falls where t = [(p-p1).(p2-p1)] / |p2-p1|^2
    -- clamp t from [0,1] to handle points outside the segment p1p2
    local t = math.max(0, math.min(1, dot(vector(line_pt1, Point), vector(line_pt1, line_pt2))/l2))
    local proj = addVector(line_pt1, scaleVector(line, t))  -- projection falls on the segment
    return len(proj, Point)
end

-- rect(A,B,C) AB vertical BC
local function insideRect(A, B, C, Point)
    local P = Point
    -- AB |_ BC
    local AB = vector(A, B)
    local AP = vector(A, P)
    local BC = vector(B, C)
    local BP = vector(B, P)

    local dotABAP = dot(AB, AP)
    local dotABAB = dot(AB, AB)
    local dotBCBP = dot(BC, BP)
    local dotBCBC = dot(BC, BC)

    local inside_flag = dotABAP>=0 and dotABAP<=dotABAB and dotBCBP >=0 and dotBCBP <= dotBCBC
    return inside_flag
end


local M = {}
local toolComponent = torch.class('toolComponent', M)

function toolComponent:__init(x1, y1, x2, y2, side_thickness)
    assert(x1 ~= x2 or y1 ~= y2)
    assert(side_thickness > 0)

    -- todo: maybe right, needs to check?
--    if x1 < x2 then
        self.p1 = {x=x1, y=y1}
        self.p2 = {x=x2, y=y2}
--    else
--        self.p1 = {x=x2, y=y2}
--        self.p2 = {x=x1, y=y1}
--    end

    self.sideThickness = side_thickness
    self.length = self:getLength()
    self.rad = self:getOrient()
    self:getSideSegments()
end

function toolComponent:inside(x,y)
    local p = {x=x,y=y }
    -- AB |_ BC
    local A = self.line1_p1
    local B = self.line2_p1
    local C = self.line2_p2

    return insideRect(A, B, C, p)
end

function toolComponent:getToolDist(x, y)
    local p = {x=x, y=y }
    return lineDist(self.p1, self.p2, p)
end

function toolComponent:getOrient()
    local dxy = vector(self.p1, self.p2)
    local rad = torch.atan2(dxy.y, dxy.x)  -- [-pi, pi]
    return rad
end

function toolComponent:getSideSegments()
    local dx = self.p2.x - self.p1.x
    local dy = self.p2.y - self.p1.y
    local dist = math.sqrt(math.pow(dx, 2) +  math.pow(dy, 2))
    dx = dx / dist
    dy = dy / dist

    local line1_x1 = self.p1.x + self.sideThickness * dy
    local line1_y1 = self.p1.y - self.sideThickness * dx
    local line2_x1 = self.p1.x - self.sideThickness * dy
    local line2_y1 = self.p1.y + self.sideThickness * dx

    local line1_x2 = self.p2.x + self.sideThickness * dy
    local line1_y2 = self.p2.y - self.sideThickness * dx
    local line2_x2 = self.p2.x - self.sideThickness * dy
    local line2_y2 = self.p2.y + self.sideThickness * dx

    self.line1_p1 = {x=line1_x1, y=line1_y1}
    self.line1_p2 = {x=line1_x2, y=line1_y2}
    self.line2_p1 = {x=line2_x1, y=line2_y1}
    self.line2_p2 = {x=line2_x2, y=line2_y2}
end

function toolComponent:getLength()
    return len(self.p1, self.p2)
end

function toolComponent:getBoundingVertices()
    local x_min, y_min, x_max, y_max
    x_min = math.min(self.line1_p1.x, self.line1_p2.x, self.line2_p1.x, self.line2_p2.x)
    x_max = math.max(self.line1_p1.x, self.line1_p2.x, self.line2_p1.x, self.line2_p2.x)
    y_min = math.min(self.line1_p1.y, self.line1_p2.y, self.line2_p1.y, self.line2_p2.y)
    y_max = math.max(self.line1_p1.y, self.line1_p2.y, self.line2_p1.y, self.line2_p2.y)
    local min_p = {x=x_min, y=y_min}
    local max_p = {x=x_max, y=y_max}
    return min_p, max_p
end

return M.toolComponent