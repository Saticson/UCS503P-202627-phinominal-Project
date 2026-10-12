package com.axiomcloud.controller;

import com.axiomcloud.service.UserService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api")
public class AuthController {

    private final UserService userService;

    public AuthController(UserService userService) {
        this.userService = userService;
    }

    @PostMapping("/auth/user")
    public ResponseEntity<Map<String, Object>> registerOrUpdateUser(
            @RequestParam("firebase_uid") String firebaseUid,
            @RequestParam("email") String email,
            @RequestParam(value = "display_name", required = false) String displayName) {
        return ResponseEntity.ok(userService.registerOrUpdateUser(firebaseUid, email, displayName));
    }

    @GetMapping("/auth/user/{firebase_uid}")
    public ResponseEntity<Map<String, Object>> getUser(@PathVariable("firebase_uid") String firebaseUid) {
        return ResponseEntity.ok(userService.getUser(firebaseUid));
    }
}
