const std = @import("std");

pub const swizzle = @import("swizzle.zig");

pub const Swizzle = swizzle.Swizzle;

pub fn Vec(comptime T: type, comptime size: u3) type {
    comptime std.debug.assert(size <= 4);
    return extern struct {
        pub const len = size;

        values: @Vector(len, T),

        pub inline fn init(values: @Vector(len, T)) @This() {
            return .{ .values = values };
        }

        pub inline fn get(self: @This(), comptime components: Swizzle) Vec(T, components.len()) {
            return .init(@shuffle(
                T,
                self.values,
                undefined,
                b: {
                    var mask: [components.len()]i32 = undefined;
                    for (components.mask(), &mask) |src, *dst| {
                        if (src >= len) @compileError("swizzle components must be smaller than self.len");
                        dst.* = src;
                    }
                    break :b mask;
                },
            ));
        }

        pub inline fn set(
            self: *@This(),
            comptime components: Swizzle,
            values: Vec(T, components.len()),
        ) void {
            comptime for (0.., components.mask()) |i, a| {
                for (components.mask()[(i + 1)..]) |b| {
                    if (a == b) {
                        @compileError("setting the same swizzle component multiple times is undefined");
                    }
                }
            };
            self.values = @shuffle(
                T,
                self.values,
                values.values,
                b: {
                    var mask: [len]i32 = undefined;
                    var i: i32 = 0;
                    while (i < mask.len) : (i += 1) {
                        var j: i32 = 0;
                        mask[i] = while (j < components.len()) : (j += 1) {
                            if (components.mask()[j] >= components.len()) {
                                @compileError("swizzle components must be smaller than values.len");
                            }
                            if (i == components.mask()[j]) {
                                break ~j;
                            }
                        } else i;
                    }
                    break :b mask;
                },
            );
        }

        pub inline fn dot(self: @This(), other: @This()) T {
            return @reduce(.Add, self.values * other.values);
        }

        pub inline fn lengthSquared(self: @This()) T {
            return dot(self, self);
        }

        pub inline fn length(self: @This()) T {
            return @sqrt(self.lengthSquared());
        }

        pub inline fn distanceSquared(self: @This(), other: @This()) T {
            return lengthSquared(.init(other.values - self.values));
        }

        pub inline fn distance(self: @This(), other: @This()) T {
            return @sqrt(distanceSquared(self, other));
        }

        pub inline fn cross(self: Vec(T, 3), other: Vec(T, 3)) Vec(T, 3) {
            var vec = self.get(.yzx).values * other.get(.zxy).values;
            vec -= self.get(.zxy).values * other.get(.yzx).values;
            return .init(vec);
        }
    };
}

test "Vec.init" {
    try std.testing.expectEqual(
        @Vector(1, i32){1},
        Vec(i32, 1).init(.{1}).values,
    );
    try std.testing.expectEqual(
        @Vector(2, i32){ 1, 2 },
        Vec(i32, 2).init(.{ 1, 2 }).values,
    );
    try std.testing.expectEqual(
        @Vector(3, i32){ 1, 2, 3 },
        Vec(i32, 3).init(.{ 1, 2, 3 }).values,
    );
    try std.testing.expectEqual(
        @Vector(4, i32){ 1, 2, 3, 4 },
        Vec(i32, 4).init(.{ 1, 2, 3, 4 }).values,
    );
    try std.testing.expectEqual(
        @Vector(4, f32){ -1.5, 2.62, 753.533, -23189531.2 },
        Vec(f32, 4).init(.{ -1.5, 2.62, 753.533, -23189531.2 }).values,
    );
}

test "Vec.get" {
    try std.testing.expectEqual(
        @Vector(1, i32){1},
        Vec(i32, 1).init(.{1}).get(.x).values,
    );
    try std.testing.expectEqual(
        @Vector(3, i32){ 1, 1, 1 },
        Vec(i32, 1).init(.{1}).get(.xxx).values,
    );
    try std.testing.expectEqual(
        @Vector(2, i32){ 2, 1 },
        Vec(i32, 2).init(.{ 1, 2 }).get(.yx).values,
    );
    try std.testing.expectEqual(
        @Vector(3, i32){ 3, 2, 3 },
        Vec(i32, 3).init(.{ 1, 2, 3 }).get(.zyz).values,
    );
    try std.testing.expectEqual(
        @Vector(4, i32){ 4, 1, 3, 2 },
        Vec(i32, 4).init(.{ 1, 2, 3, 4 }).get(.wxzy).values,
    );
}

test "Vec.set" {
    {
        var vec = Vec(i32, 1).init(@splat(0));
        vec.set(.x, Vec(i32, 1).init(.{1}));
        try std.testing.expectEqual(
            @Vector(1, i32){1},
            vec.values,
        );
    }
    {
        var vec = Vec(i32, 2).init(@splat(0));
        vec.set(.xy, Vec(i32, 2).init(.{ 1, 2 }));
        try std.testing.expectEqual(
            @Vector(2, i32){ 1, 2 },
            vec.values,
        );
    }
    {
        var vec = Vec(i32, 3).init(@splat(0));
        vec.set(.zxy, Vec(i32, 3).init(.{ 1, 2, 3 }));
        try std.testing.expectEqual(
            @Vector(3, i32){ 2, 3, 1 },
            vec.values,
        );
    }
    {
        var vec = Vec(i32, 4).init(@splat(0));
        vec.set(.zxwy, Vec(i32, 3).init(.{ 1, 2, 3 }).get(.zyxz));
        try std.testing.expectEqual(
            @Vector(4, i32){ 2, 3, 3, 1 },
            vec.values,
        );
    }
}

test "Vec.dot" {
    try std.testing.expectEqual(
        92,
        Vec(i32, 2)
            .init(.{ 12, 20 })
            .dot(Vec(i32, 2).init(.{ 16, -5 })),
    );
    try std.testing.expectApproxEqAbs(
        -2913.84,
        Vec(f32, 3)
            .init(.{ 64.15, 7.8734, -213.53 })
            .dot(Vec(f32, 3).init(.{ 0.2, -127.6326, 9 })),
        0.01,
    );
}

test "Vec.cross" {
    {
        const tolerance: @Vector(3, f32) = @splat(1e-5);
        const expected = @Vector(3, f32){ -2.0 / 3.0, 1.0 / 2.0, 5.0 / 12.0 };
        const a = Vec(f32, 3).init(.{ 1.0 / 4.0, -1.0 / 2.0, 1.0 });
        const b = Vec(f32, 3).init(.{ 1.0 / 3.0, 1.0, -2.0 / 3.0 });
        const actual = a.cross(b).values;
        if (!@reduce(.And, @abs(expected - actual) < tolerance)) {
            std.debug.print("actual {}, not within absolute tolerance {} of expected {}\n", .{
                actual,
                tolerance,
                expected,
            });
            return error.TestExpectedApproxEqAbs;
        }
    }
    {
        const expected = @Vector(3, i32){ -26988, -1002, -8142 };
        const a = Vec(i32, 3).init(.{ 64, 7, -213 });
        const b = Vec(i32, 3).init(.{ 2, -127, 9 });
        const actual = a.cross(b).values;
        try std.testing.expectEqual(expected, actual);
    }
}
