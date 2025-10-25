const std = @import("std");

pub const vector = @import("vector.zig");

pub const Vec = vector.Vec;

test {
    std.testing.refAllDeclsRecursive(@This());
}
