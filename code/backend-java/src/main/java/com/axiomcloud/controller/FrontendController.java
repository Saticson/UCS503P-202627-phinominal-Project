package com.axiomcloud.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class FrontendController {

    @GetMapping({"/", "/index.html"})
    public String index() {
        return "index";
    }

    @GetMapping("/dashboard")
    public String dashboard() {
        return "dashboard";
    }

    @GetMapping("/datasets")
    public String datasets() {
        return "datasets";
    }

    @GetMapping("/training")
    public String training() {
        return "training";
    }

    @GetMapping("/models")
    public String models() {
        return "models";
    }

    @GetMapping("/predict")
    public String predict() {
        return "predict";
    }
}
